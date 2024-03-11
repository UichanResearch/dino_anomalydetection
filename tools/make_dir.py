import os

class _PathError(Exception):
    def __init__(self, message="PathError"):
        self.message = message
        super().__init__(self.message)

def make_dir(name = "default", root = "result"):
    target = os.path.join(root,name)

    if os.path.exists(target):
        raise _PathError("project name is already exist")
    
    os.mkdir(target)
    os.mkdir(os.path.join(target,"model"))
    os.mkdir(os.path.join(target,"imgs"))