local ConstrainedDiv, parent = torch.class('nn.ConstrainedDiv', 'nn.Module')

function ConstrainedDiv:__init(inplace)
   parent.__init(self)
   
   self.weight = torch.ones(1)
   self.gradWeight = torch.zeros(1)
   self.inplace = inplace or false

   self:reset()
end


function ConstrainedDiv:reset(stdv)
   self.weight:fill(1);
end

function ConstrainedDiv:updateOutput(input)
   if self.weight[1] <= 0.1 then
      self.weight[1] = 0.11
   end
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input);
   end
   self.output:mul(1/self.weight[1]);
   return self.output
end

function ConstrainedDiv:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput = gradOutput
      self.gradInput:add(1/self.weight[1])
   else
      self.gradInput:resizeAs(input):zero()
      self.gradInput:add(1/self.weight[1], gradOutput)
   end
   return self.gradInput
end

function ConstrainedDiv:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput)*(-1/(self.weight[1]^2));
end
