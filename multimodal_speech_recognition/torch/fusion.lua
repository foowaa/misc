local nn = require 'nn'

local Model = {}

function Model:createFusion(X)
  local featureSize =  X:size(2)

  -- Create encoder
  self.encoder = nn.Sequential()
  --self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, 150))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(150, 100))
  self.encoder:add(nn.Sigmoid(true))
  --self.encoder:add(nn.Linear(64, 32))
  --self.encoder:add(nn.Sigmoid(true))

  -- Create decoder
  self.decoder = nn.Sequential()
  --self.decoder:add(nn.Linear(32, 64))
  --self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(100, 150))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(150, featureSize))
  self.decoder:add(nn.Sigmoid(true))
  --self.decoder:add(nn.View(X:size(2), X:size(3)))

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

return Model
