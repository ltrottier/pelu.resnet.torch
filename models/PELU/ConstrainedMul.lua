local ConstrainedMul, parent = torch.class('nn.ConstrainedMul', 'nn.Module')

function ConstrainedMul:__init(inplace)
   parent.__init(self)
   
   self.weight = torch.ones(1)
   self.gradWeight = torch.zeros(1)
   self.inplace = inplace or false

   self:reset()
end


function ConstrainedMul:reset(stdv)
   self.weight:fill(1);
end

function ConstrainedMul:updateOutput(input)
   if self.weight[1] >= 2 then
      self.weight[1] = 1.99
   end
   if self.weight[1] <= 0.1 then
      self.weight[1] = 0.11
   end
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input);
   end
   self.output:mul(self.weight[1]);
   return self.output
end

function ConstrainedMul:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput = gradOutput
      self.gradInput:add(self.weight[1])
   else
      self.gradInput:resizeAs(input):zero()
      self.gradInput:add(self.weight[1], gradOutput)
   end
   return self.gradInput
end

function ConstrainedMul:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);
end
