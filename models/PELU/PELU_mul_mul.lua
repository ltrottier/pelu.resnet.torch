require 'nn'

--pcall(require, './ConstrainedDiv')
--pcall(require, './ConstrainedMul')

local function PELU()
    local pelu = nn.Sequential()
    pelu:add(nn.ConstrainedMul(false))
    pelu:add(nn.ELU(1, true))
    pelu:add(nn.ConstrainedMul(false))

    return pelu
end

return PELU
