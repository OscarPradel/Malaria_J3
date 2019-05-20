def Versions():    
    import sys
    print(f'{"Python version:":24s} {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.')
    try:
        import sklearn as skl 
        print(f'{"Scikit-learn version:":24s} {skl.__version__}.')
    except:
        print(f'WARN: could not find sklearn!')  
    try:
        import keras as kr
        print(f'{"Keras version:":24s} {kr.__version__}')
    except:
        print(f'WARN: could not find keras!')  
    try:
        import tensorflow as tf
        print(f'{"Tensorflow version:":24s} {tf.__version__}')
    except:
        print(f'WARN: could not find tensorflow!')  

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
    if K_gpu:
        raise ImportException("no keras import allowed before calling DisableGPU(), reset your kernel and call DisableGPU() in the beginning of your code.")
         
def RestartKernel() :
    try:
        from IPython.display import display_html
        display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
    except:
         print("ERROR in RestartKernel()!")