def UsesGPU():    
    TF_gpu = False
    K_gpu  = False    
    
    try:
        # confirm TensorFlow sees the GPU
        from tensorflow.python.client import device_lib
        if 'GPU' in str(device_lib.list_local_devices()):
            TF_gpu = True      
    except:
        print("WARNING: could not import from tensorflow"); 
              
    try:
        # confirm Keras sees the GPU
        from keras import backend
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            K_gpu = True
    except:
        print("WARNING: could not import from keras");
    
    return TF_gpu, K_gpu

def DisableGPU():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    TF_gpu, K_gpu = UsesGPU()
    #if K_gpu:
    #    raise ImportException("no keras import allowed before calling DisableGPU(), reset your kernel and call DisableGPU() in the beginning of your code.")

def EnableGPU(gpu_mem_fraction=0.05, gpus=None, cpus=None):
    #try:
    #    import os
    #    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #except:
    #    print("ERROR: something failed in EnableGPU(), environment part")
    
    try:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()        
        if gpus!=None or cpus!=None:
            if gpus==None:
                gpus=1 # default
            if cpus==None:
                cpus=4 # default
            config = tf.ConfigProto( device_count = { 'GPU': gpus, 'CPU': cpus}  )
            if gpu_mem_fraction<0:
                raise ValueExeption("gpu_mem_fraction is below zero")
            if gpu_mem_fraction>0.5:
                raise ValueExeption("sorry Dave, can't do gpu_mem_fraction>50%")
        config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
        config.gpu_options.allow_growth=True
        set_session(tf.Session(config=config))
    except:
        print("ERROR: something failed in EnableGPU(), Tensorflow part")
              
def RestartKernel() :
    try:
        from IPython.display import display_html
        display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
    except:
         print("ERROR in RestartKernel()!")