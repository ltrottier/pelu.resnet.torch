local PELU, Parent = torch.class('nn.PELU', 'nn.Module')


function PELU:__init()
    Parent.__init(self)
    self.train = true
    self.weight = torch.ones(2)
    self.gradWeight = torch.zeros(2)
end

function PELU:updateOutput(input)
    self.output:resizeAs(input):zero()
    
    if self.weight[1] >= 2 then
        self.weight[1] = 1.99
    end
    if self.weight[1] <= 0.1 then
        self.weight[1] = 0.11
    end
    local a = self.weight[1]
    if self.weight[2] <= 0.1 then
        self.weight[2] = 0.11
    end
    local b = self.weight[2]
    
    self.output[input:lt(0)] = input[input:lt(0)]:div(b):exp():add(-1):mul(a)    
    self.output[input:ge(0)] = input[input:ge(0)]*a/b
    
    --self.output = nn.ELU(1):forward(input / b) * a
    
    collectgarbage()
    
    return self.output
end

function PELU:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput):zero()
    local a = self.weight[1]
    local b = self.weight[2]
    
    self.gradInput[input:lt(0)] = input[input:lt(0)]:div(b):exp():mul(a):div(b)
    self.gradInput[input:ge(0)] = a/b
    
    self.gradInput:cmul(gradOutput)
    
    return self.gradInput
end

function PELU:accGradParameters(input, gradOutput, scale)
    self.gradWeight:resizeAs(self.weight):zero()
    local a = self.weight[1]
    local b = self.weight[2]
    local scale = scale or 1
   -- for a
    self.gradWeight[{ {1} }]:add( input[input:lt(0)]:div(b):exp():add(-1):cmul(gradOutput[input:lt(0)]):sum() )
    self.gradWeight[{ {1} }]:add( input[input:ge(0)]:div(b):cmul(gradOutput[input:ge(0)]):sum() )
    self.gradWeight[{ {1} }]:mul(scale)
   
    --local gradA = torch.cmul(torch.exp(input/b) - 1, input:lt(0):type(input:type()) ) + torch.cmul(input/b, input:ge(0):type(input:type())) 
    --self.gradWeight[{ {1} }] = torch.cmul(gradA, gradOutput):sum() * scale
    
    -- for b
    self.gradWeight[{ {2} }]:add( input[input:lt(0)]:div(b):exp():cmul(input[input:lt(0)]):mul(-a):div(b^2):cmul(gradOutput[input:lt(0)]):sum() )
    self.gradWeight[{ {2} }]:add( input[input:ge(0)]:mul(-a):div(b^2):cmul(gradOutput[input:ge(0)]):sum() )
    self.gradWeight[{ {2} }]:mul(scale)
    
    --local gradB = torch.cmul(torch.exp(input/b):cmul(input) * (-a) / b^2, input:lt(0):type(input:type()) ) + torch.cmul(input * (-a) / b^2, input:ge(0):type(input:type())) 
    -- self.gradWeight[{ {2} }] = torch.cmul(gradB, gradOutput):sum() * scale
    
    collectgarbage()
    --print(a,b)
end


function PELU:__tostring__()
   return string.format('%s', torch.type(self))
end

