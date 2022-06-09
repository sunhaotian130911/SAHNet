class DefaultConfig(object):
    mulitGPU = False
    debug = False
    train_data_root = ''
    validation_data_root = ''
    test_data_root = ''
    load_model_path = ''
    batch_size = 4
    use_gpu = True
    num_workers = 8

    max_epoch = 40
    lr = 0.0001
    weight_decay = 0.0001  # 0.01

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" %k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
