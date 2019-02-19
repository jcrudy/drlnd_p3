import os
from toolz import curry
import platform
import urllib.request
import zipfile
from zipfile import ZipFile
from ..util import split_path

# def match_filename(directory, pattern):
#     contents = list(filter(curry(flip(fnmatchcase))(pattern), os.listdir(directory)))
#     if len(contents) > 1:
#         raise ValueError('File pattern is ambiguous.')
#     if not contents:
#         return None
#     return contents[0]

def unzip_and_fix_permissions(zippath, directory, rename=None):
    '''
    Unzip with all permissions set to default.  This is needed because 
    the permissions in Banana.app are not set correctly as archived, causing
    the environment to fail to load.  It's necessary to do it this way so the
    program doesn't need to be run with sudo.
    
    Modified from https://stackoverflow.com/a/596455/1572508.
    '''
    infile = ZipFile(zippath, 'r')
    for file in infile.filelist:
        name = file.filename
        path_list = split_path(name)
        if rename is not None:
            path_list[0] = rename
        if name.endswith('/'):
            path_list[-1] += '/'
            outfile = os.path.join(directory, *path_list)
            try:
                os.mkdir(outfile)
            except FileExistsError:
                pass
        else:
            outfile = os.path.join(directory, *path_list)
            fh = os.open(outfile, (os.O_CREAT | os.O_WRONLY))
            os.write(fh, infile.read(name))
            os.close(fh)
            os.chmod(outfile, 0o777)

@curry
def download_unity_environment(linux_url, mac_url, win32_url, win64_url, 
                               directory, linux_name=None, mac_name=None, 
                               win32_name=None, win64_name=None):
    '''
    Download the correct compiled environment for this platform, extract the zip archive, 
    and fix all file permissions.
    '''
    system = platform.system()
    machine = platform.machine()
    if system == 'Linux':
        url = linux_url
        name = linux_name
    elif system == 'Darwin':
        url = mac_url
        name = mac_name
    elif system == 'Windows':
        if machine == 'AMD64' or machine == 'x86_64':
            url = win64_url
            name = win64_name
        else:
            url = win32_url
            name = win32_name
    path = os.path.join(directory, url.split('/')[-1])
    if name is None:
        name = '.'.join(os.path.split(path)[-1].split('.')[:-1])
    result = os.path.join(directory, name)
    if not os.path.exists(result):
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('Downloading {}...'.format(url), end='')
        urllib.request.urlretrieve(url, path) 
        print('Download complete.')
        print('Extracting {}...'.format(path), end='')
        try:
            unzip_and_fix_permissions(path, directory, rename=name)
        except:
            raise
            # Possibly on some systems the above will both fail and 
            # be unnecessary.  Then do it this way, which will not 
            # attempt to alter permissions.
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(directory)
        os.remove(path)
        print('Extraction complete.')
    
    return result

resources = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')

banana = download_unity_environment(
                linux_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip',
                mac_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip',
                win32_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip',
                win64_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip',
                directory = os.path.join(resources, 'banana')
                )

reacher_v1 = download_unity_environment(
                linux_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip',
                mac_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip',
                win32_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip',
                win64_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip',
                directory = os.path.join(resources, 'reacher_v1')
                )

reacher_v2 = download_unity_environment(
                linux_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip',
                mac_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip',
                win32_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip',
                win64_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip',
                directory = os.path.join(resources, 'reacher_v2')
                )


tennis = download_unity_environment(
                linux_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip',
                mac_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip',
                win32_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip',
                win64_url='https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip',
                directory = os.path.join(resources, 'tennis'),
                linux_name=os.path.join('Tennis_Linux', 'Tennis.x86_64')
                )

