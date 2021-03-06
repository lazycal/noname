# Copyright 2019 Bytedance Inc. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import noname.nonametorch.ops
from noname.nonametorch.ops import push_pull_async_inplace as noname_push_pull
# from noname.nonametorch.ops import push_pull
from noname.nonametorch.ops import synchronize, declare, declare_done
from noname.nonametorch.ops import init, shutdown
from noname.nonametorch.ops import size, local_size, rank, local_rank
from noname.nonametorch.compression import Compression

import os
import torch
import collections

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, module, named_parameters, compression,
                 backward_passes_per_step=1):
        super(self.__class__, self).__init__(params)
        self._compression = compression
        self._module = module
        self._bn_named_param, self._bn_modules = self._get_bn_named_param()

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._enable_async = False #(int(os.getenv('BYTEPS_ENABLE_ASYNC', 0)) != 0)
        if self._enable_async:
            assert int(os.getenv('DMLC_NUM_WORKER')) > 1, \
                "Async is only valid for distributed training"
            print('BytePS: enable asynchronous training')

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = _DistributedOptimizer.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        if len(named_parameters) > 0:
            if isinstance(named_parameters[0][1], torch.Tensor):
                if any([not isinstance(p, torch.Tensor) for name, p in named_parameters]):
                    raise ValueError('named_parameters should consistently be a sequence of '
                                     'tuples (name, torch.Tensor)')
                self._is_tensor_instance = True
                # there is an issue when using torch.Tensor as key, so use its hash instead
                # https://github.com/pytorch/pytorch/issues/7733
                named_parameters.extend(list(self._bn_named_param.items()))
                # print(named_parameters)
                self._parameter_names = {v.__hash__(): k for k, v
                                         in sorted(named_parameters)}
                # self._tensor_list = [tensor for name, tensor in named_parameters]
            else:
                raise NotImplementedError()
                self._is_tensor_instance = False
                self._parameter_names = {v: k for k, v
                                         in sorted(named_parameters)}
        else:
            raise NotImplementedError()
            self._is_tensor_instance = False
            self._parameter_names = {v: 'push_pull.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        self.backward_passes_per_step = backward_passes_per_step
        self._push_pull_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._named_parameters = dict(named_parameters)
        if size() > 1:
            self._register_hooks()

            # declare tensors
            for idx, name in enumerate(sorted(self._parameter_names.values())):
                p = self._named_parameters[name]
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    declare("Gradient."+name, idx, p.element_size() * p.nelement())
                    # handle = noname_push_pull(p.grad, average=False, name="Gradient."+name)
            # We use two loops for load-balancing
            # for name in sorted(self._bn_named_param.keys()):
            #     declare("Parameter."+name)
            declare_done()
    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._push_pull_delay:
            self._push_pull_delay[p] = self.backward_passes_per_step

    def _get_bn_named_param(self):
        res = {}
        bn_modules = {}
        cnt = 0
        for module in self._module.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                res[f'BN{cnt}.running_mean'] = module.running_mean
                res[f'BN{cnt}.running_var'] = module.running_var
                bn_modules[f'BN{cnt}.running_mean'] = module
                bn_modules[f'BN{cnt}.running_var'] = module
                cnt += 1
        # print(res)
        return res, bn_modules

    def _register_bn_hooks(self):
        for name, p in self._bn_named_param.items():
            self._bn_modules[name].register_forward_hook(
                self._make_hook(p, prefix='Parameter.'))

    def _register_hooks(self):
        # self._register_bn_hooks()
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _push_pull_grad_async(self, p, prefix="Gradient."):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        if self._enable_async:
            # the real handle will be created in step()
            handle, ctx = None, None
        else:
            tensor = p.grad if prefix == 'Gradient.' else p
            tensor_compressed, ctx = self._compression.compress(tensor)
            handle = noname_push_pull(tensor_compressed, average=True, name=prefix+name)
        return handle, ctx

    def _make_hook(self, p, prefix="Gradient."):
        def hook(module, *ignore):
            if isinstance(module, torch.nn.BatchNorm2d):
                if not module.training: return
            if p in self._handles and self._handles[p][0] is not None:
                if self._push_pull_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert self._push_pull_delay[p] > 0
            handle, ctx = None, None
            self._push_pull_delay[p] -= 1
            if self._push_pull_delay[p] == 0:
                handle, ctx = self._push_pull_grad_async(p, prefix=prefix)
            self._handles[p] = (handle, ctx)
        return hook

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = self._push_pull_grad_async(p)
            self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx = self._push_pull_grad_async(p)
                self._handles[p] = (handle, ctx)
        for p, (handle, _) in self._handles.items():
            synchronize(handle)
            self._push_pull_delay[p] = self.backward_passes_per_step
            # if not self._enable_async:
            #     p.grad.set_(self._compression.decompress(output, ctx))
        self._handles.clear()

    def step(self, closure=None):
        # if self._enable_async:
        #     old_weight_map = {}
        #     # store the weights before update
        #     for p, _ in self._handles.items():
        #         old_weight_map[p] = p.data.clone().detach()
        #     # update
        #     loss = super(self.__class__, self).step(closure)
            
        #     for p, (h, _) in self._handles.items():
        #         # get the diff for each weight (in-place)
        #         p.data.sub_(old_weight_map.get(p))
        #         if h is None:
        #             # create the handler now
        #             if self._is_tensor_instance:
        #                 name = self._parameter_names.get(p.__hash__())
        #             else:
        #                 name = self._parameter_names.get(p)
        #             handle = noname_push_pull(p, average=False, name="AsyncParam."+name)
        #             _, ctx = self._compression.compress(p)
        #             self._handles[p] = (handle, ctx)

        #     self.synchronize()
        #     noname.nonametorch.ops.step()
        #     return loss
        # else:
            self.synchronize()
            noname.nonametorch.ops.step()
            return super(self.__class__, self).step(closure)


def DistributedOptimizer(optimizer, module, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an push_pull to
    average gradient values before applying gradients to model weights.
    push_pull operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all push_pull operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces push_pull operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.
    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          push_pull operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during push_pull to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an push_pull implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, module, named_parameters,
               compression, backward_passes_per_step)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.
    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    raise NotImplementedError()
    # if isinstance(params, dict):
    #     params = sorted(params.items())
    # elif isinstance(params, list):
    #     # support both named_parameters() and regular parameters()
    #     params = [p if isinstance(p, tuple) else (None, p) for p in params]
    # else:
    #     raise ValueError('invalid params of type: %s' % type(params))

    # # Run synchronous broadcasts.
    # for name, p in params:
    #     # Broadcast is implemented as push + pull in BytePS
    #     # To make it a real broadcast, we set the non-root tensors all 0.
    #     if rank() != root_rank:
    #         p.fill_(0)
    #     # Remember to disable averaging because we are doing broadcast
    #     handle = noname_push_pull(p, average=False, name="Parameter."+name)
    #     synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.
    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    raise NotImplementedError()
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces push_pull on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(
                option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
