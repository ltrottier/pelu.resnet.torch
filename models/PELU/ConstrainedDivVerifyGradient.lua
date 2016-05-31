require 'nn'
require 'ConstrainedDiv'
y = torch.Tensor(5)
mlp = nn.Sequential()
--mlp:add(nn.Copy())
mlp:add(nn.ConstrainedDiv())
--mlp:add(nn.Copy())

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   return err
end

for i = 1, 10000 do
   x = torch.rand(5)
   y:copy(x)
   y:mul(math.pi)
   err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end

print(mlp:get(1).weight:pow(-1))
