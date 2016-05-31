
require 'nn'
require 'gnuplot'
require 'optim'
require 'PELU'
print '==> testing backprop with Jacobian (finite element)'

require 'mattorch'

-- define inputs and module
input = torch.linspace(0,1,100)
input = torch.cat(input, -torch.linspace(0,1,100),2)
input = torch.cat(input, torch.linspace(0,1,100):mul(2):add(-1),2)

input = input + torch.randn(100,input:size(2))*0.01 + 5

input = torch.add(input, -torch.mean(input,1):expand(input:size()))

--input = torch.Tensor(100,2)
--input[{ {},{1} }] = torch.linspace(0,1,100)
--input[{ {},{2} }] = torch.linspace(1,2,100)

inputSize = input:size()



-- Input - Output gradient
module = nn.PELU()
function feval(x)
    x = x:reshape(inputSize)
    module:forward(x)
    module:backward(x,torch.ones(x:size()))
    --print(module.criterionValue)
    return module.output:sum(), module.gradInput:reshape(x:numel())
end


diff, dc, dc_est = optim.checkgrad(feval, input:reshape(input:numel()))
print('Diff must be close to 1e-8: diff = ' .. diff)



-- Parameters gradient
module = nn.PELU()
function feval2(w)
    module.weight = w:clone()
    module:forward(input)
    module:backward(input,torch.ones(input:size()))
    return module.output:sum(), module.gradWeight
end


diff, dc, dc_est = optim.checkgrad(feval2, torch.ones(module.weight:size()) * 0.5 )
print('Diff must be close to 1e-8: diff = ' .. diff)

