local URL = "http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"

local M = {}

local function convertBinaryToTensor(inputFile, outputFile)
    local m = torch.DiskFile(inputFile, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamples = length / 3074
    assert(nSamples == math.floor(nSamples), 'expecting numSamples to be an exact integer')
    m:seek(1)

    local coarse = torch.ByteTensor(nSamples)
    local fine = torch.ByteTensor(nSamples)
    local data = torch.ByteTensor(nSamples, 3, 32, 32)

    for i=1,nSamples do
        coarse[i] = m:readByte()
        fine[i]   = m:readByte()
        local store = m:readByte(3072)
        data[i]:copy(torch.ByteTensor(store))
    end

    fine:add(1)

    local out = {}
    out.data = data
    out.labels = fine
    out.labelCoarse = coarse
    torch.save(outputFile, out)
    return out
end

function M.exec(opt, cacheFile)
   print("=> Downloading CIFAR-100 dataset from " .. URL)
   local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
   assert(ok == true or ok == 0, 'error downloading CIFAR-100')

   print(" | converting binary file to tensor")
   local trainData = convertBinaryToTensor('gen/cifar-100-binary/train.bin', 'gen/cifar100-train.t7')
   local testData = convertBinaryToTensor('gen/cifar-100-binary/test.bin', 'gen/cifar100-test.t7')

   print(" | saving CIFAR-10 dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

