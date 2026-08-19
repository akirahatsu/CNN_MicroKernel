class CNN_OLD:

  def __init__(self ,kernel_shape = (2,2) , n_kernel = 1):

    self.kernel_shape = kernel_shape
    self.n_kernel = n_kernel

    self.bias = np.zeros((n_kernel))


  def forward(self , input):

    self.input = np.array(input)


    self.batch ,self.channel , self.height , self.width = self.input.shape

    n_in = self.channel * self.kernel_shape[0] * self.kernel_shape[1]

    if not hasattr(self, "weight"):
      self.weight = (np.random.randn(
          self.n_kernel,
          self.channel,
          *self.kernel_shape
      ) *  np.sqrt(2 / n_in))




    self.out_h = self.height - self.weight.shape[2] + 1
    self.out_w = self.width - self.weight.shape[3] + 1

    self.output = np.zeros((self.batch , self.n_kernel, self.out_h , self.out_w))

    for b in range(self.batch):
      for k in range(self.n_kernel):
        for h in range(self.out_h):
          for w in range(self.out_w):

            total = 0

            for c in range(self.channel):
              patch = self.input[b,c,

                                h : h + self.weight.shape[2],
                                w : w + self.weight.shape[3]
                                ]
              total += np.sum(patch * self.weight[k,c])




            self.output[b, k, h , w ] = total + self.bias[k]


    return   self.output

  def backward(self,dl_dc):

    self.dl_dw = np.zeros_like(self.weight)
    self.dl_db = np.zeros_like(self.bias)
    self.dl_di = np.zeros_like(self.input ,dtype= float)


    for b in range(self.batch):
      for k in range(self.n_kernel):
        for h in range(self.out_h):
          for w in range(self.out_w):

            for c in range(self.channel):

              patch = self.input[b,c,

                                h : h + self.weight.shape[2],
                                w : w + self.weight.shape[3]
                                ]
              self.dl_dw[k,c] += patch * dl_dc[b,k,h,w]

              self.dl_db[k] += dl_dc[b,k,h,w]

              self.dl_di[b,c,
                    h : h + self.weight.shape[2],
                    w : w + self.weight.shape[3]] += self.weight[k,c] * dl_dc[b,k,h,w]

    return self.dl_di
