from os import path as ospath

def relpath(*args):
    root = ospath.dirname(__file__)
    if isinstance(args, str):
        return ospath.join(root, '..', args)
    return ospath.join(root, '..', *args)
