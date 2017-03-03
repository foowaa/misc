local nn = require 'nn'
require 'rnn'
require 'nngraph'

local Model = {}

function Model:createLSTM( X )
  local featureSize =  X:size(2)
  local classes = 10
  X = X:contiguous()
  --self.lstm = 
  self.lstm = nn.Sequential()
  nn.FastLSTM.usernngraph = true
  nn.FastLSTM.bn = false
  --self.lstm:add(nn.Sequencer(nn.LookupTable(featureSize, 100)))
  --local rnn = nn.LSTM(featureSize,100)
  self.lstm:add(nn.Sequencer(nn.FastLSTM(featureSize, 64)))
  self.lstm:add(nn.Sequencer(nn.Linear(64,32)))
  self.lstm:add(nn.Sequencer(nn.ReLU(true)))
  self.lstm:add(nn.Sequencer(nn.Linear(32,classes)))
  self.lstm:add(nn.Sequencer(nn.LogSoftMax()))
end

return Model



-- local nn = require 'nn'
-- require 'rnn'
-- require 'nngraph'

-- local Model = {}

-- function Model:createLSTM( X )
--   local featureSize =  X:size(2)
--   local classes = 10
--   local hiddenSize = featureSize
--   local rho = 40
--   X = X:contiguous()
--   --self.lstm = 
--   self.rnn = nn.Sequential()
--   local r = nn.Recurrent(
--         hiddenSize, nn.Identity(),
--         nn.Linear(hiddenSize, hiddenSize), nn.ReLU(),
--         rho
--     )
--   self.rnn:add(nn.SplitTable(1))
--  self.rnn:add(nn.Sequencer(r))
--  self.rnn:add(nn.SelectTable(-1))
--  self.rnn:add(nn.Sequencer(nn.Linear(hiddenSize, classes)))
--  self.rnn:add(nn.Sequencer(nn.LogSoftMax()))
--   --self.lstm:add(nn.Sequencer(nn.LookupTable(featureSize, 100)))
--   --local rnn = nn.LSTM(featureSize,100)

-- end

-- return Model


