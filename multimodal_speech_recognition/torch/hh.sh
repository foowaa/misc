#!/bin/bash
echo '------------------------------'
echo 'Training Start'
th trainVAE.lua
th trainFusion.lua
python lstm.lua
echo '------------------------------'
echo 'Training End'
