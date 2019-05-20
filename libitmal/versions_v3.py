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
