require 'nn'

require './ConstrainedDiv'
require './ConstrainedMul'

function PELU()
    local pelu = nn.Sequential()
    pelu:add(nn.ConstrainedDiv(false))
    pelu:add(nn.ELU(1, true))
    pelu:add(nn.ConstrainedMul(false))

    return pelu
end

return PELU
