import cProfile

def dask_setup(worker):
    print("dask setup HAPPENING")
    worker.profile = cProfile.Profile()
    worker.profile.enable()
