class CNN:

  def __init__(self , kernel_size  , n_kernel):
    self.kernel_size = kernel_size
    self.n_kernel = n_kernel
    self.bias =  np.zeros(self.n_kernel)

    if self.kernel_size[0] == 2:
      self.hard_patch = lambda x: np.stack([
                x[..., :-1, :-1], x[..., :-1, 1:],
                x[..., 1:, :-1],  x[..., 1:, 1:]
            ], axis=2)
      # self.hard_patch = lambda input: np.array([
      #                          input[:-1, :-1], input[:-1, 1:], 
      #                          input[1:, :-1], input[1:, 1:]
      #                      ])
    elif self.kernel_size[0] == 3:

      self.hard_patch = lambda x: np.stack([
                x[..., :-2, :-2], x[..., :-2, 1:-1], x[..., :-2, 2:],
                x[..., 1:-1, :-2], x[..., 1:-1, 1:-1], x[..., 1:-1, 2:],
                x[..., 2:, :-2],  x[..., 2:, 1:-1],  x[..., 2:, 2:]
            ], axis=2)
      
      # self.hard_patch = lambda input : np.array([input[:-2, :-2] , input[:-2, 1:-1] , input[:-2, 2:]  ,
      #                             input[1:-1, :-2] ,input[1:-1, 1:-1] , input[1:-1, 2:] ,
      #                             input[2:, :-2] ,input[2:, 1:-1] ,input[2:, 2:]])


  def hard_cnn(self, patches, kernel):
    return patches * kernel[:, None, None]



  def forward(self , input):

    self.input = np.array(input)
    self.batch , self.channel , self.height ,self.width = self.input.shape

    if not hasattr(self,'kernel'):

      n_in = self.channel * self.kernel_size[0] * self.kernel_size[1]

      self.base_kernel = (np.random.randn(self.n_kernel ,
                                     self.channel ,
                                     *self.kernel_size) *  np.sqrt(2 / n_in))

      self.kernel = self.base_kernel.reshape(self.n_kernel,self.channel,-1)

    out_h = self.height - self.kernel_size[0] + 1
    out_w = self.width - self.kernel_size[1] + 1

    self.output = np.zeros(
            (self.batch, self.n_kernel, out_h, out_w),
            dtype=np.float32)


    patches = self.hard_patch(self.input)

    total = np.einsum("bcphw,kcp->bkhw", patches, self.kernel)
    self.output = total + self.bias[None, :, None, None]

    # for b in range(self.batch):
    #   for k in range(self.n_kernel):
    #     total = 0
    #     for c in range(self.channel):
    #       patches = self.hard_patch(self.input[b, c])

    #       total += np.sum(patches * self.kernel[k,c][:, None, None],axis = 0)
    #     self.output[b,k] = total + self.bias[k]
    return self.output



  def backward(self, dl_dcnn):

    self.dl_dw = np.zeros_like(self.kernel)
    self.dl_db = np.zeros_like(self.bias)
    self.dl_di = np.zeros_like(self.input, dtype=float)


    self.dl_db = np.einsum("bkhw -> k " , dl_dcnn)
    all_patches = self.hard_patch(self.input)

    self.dl_dw = np.einsum(
        "bcphw,bkhw->kcp",
        all_patches,
        dl_dcnn
    )

    pad_h = self.kernel_size[0] - 1
    pad_w = self.kernel_size[1] - 1

    padded_dl = np.pad(
        dl_dcnn,
        (
            (0, 0),              # batch
            (0, 0),              # kernels
            (pad_h, pad_h),      # height
            (pad_w, pad_w)       # width
        ),
        mode='constant'
    )
    dl_di_patches = self.hard_patch(padded_dl)



    rotated = np.rot90(
    self.base_kernel,
    2,
    axes=(2, 3)
     )

    self.dl_di = np.einsum('bkphw,kcp->bchw' , dl_di_patches ,rotated.reshape(self.n_kernel, self.channel, -1))


    # for b in range(self.batch):
    #   for k in range(self.n_kernel):

    #     pad_h = self.kernel_size[0] - 1
    #     pad_w = self.kernel_size[1] - 1

    #     padded_dl = np.pad(
    #           dl_dcnn[b, k],
    #           ((pad_h, pad_h), (pad_w, pad_w)),
    #           mode='constant'
    #       )

    #     dl_di_patches = self.hard_patch(padded_dl)

    #     self.dl_db[k] += np.sum(dl_dcnn[b, k])

    #     for c in range(self.channel):

    #       patches = self.hard_patch(self.input[b, c])


    #       # self.dl_dw[k, c] += np.sum(patches * dl_dcnn[b, k][None, :, :],axis=(1, 2))

    #       kernel = self.base_kernel[k, c]
    #       rotated = np.rot90(kernel, 2).reshape(-1)

    #       self.dl_di[b,c] += np.sum(
    #           dl_di_patches * rotated[:, None, None],
    #           axis=0
    #       )
    # self.dl_di()

    self.dl_dw = self.dl_dw.reshape(self.base_kernel.shape)

    return self.dl_dw
