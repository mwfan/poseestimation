from finalproject.storagemanager import StorageManager

sm = StorageManager()

def send_images_example(vid1, vid2):
    public_urls = []
    # get the list of file store in the vid1 and vid2 directories
    # vid1 = os.listdir(r'C:\Users\19494\Downloads\Dori\Videos\vid1')
    # vid2 = os.listdir(r'C:\Users\19494\Downloads\Dori\Videos\vid2')

    public_url1 = sm.upload_file(file_name='dance/{vid1}', local_path=vid1)
    public_url2 = sm.upload_file(file_name='dance/{vid2}', local_path=vid2)
    public_urls.append((public_url1, public_url2))
    print(public_urls)
    return public_urls

#send_images_example('/Users/sarahmw/PycharmProjects/project/finalproject/shorter.mov', '/Users/sarahmw/PycharmProjects/project/finalproject/RPReplay_Final1658705158.mov')


