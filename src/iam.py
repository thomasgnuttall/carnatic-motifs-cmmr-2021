def unpack_saraga_metadata(metadata, svara_freq):
    raaga = metadata.get('raaga', None)
    if raaga:
        raaga = raaga[0].get('name', None)
        
    taala = metadata.get('taala', None)
    if taala:
        taala = taala[0].get('name', None)
        
    album_artists = metadata.get('album_artists', None)
    if album_artists:
        album_artists = album_artists[0].get('name', None)
        
    concert = metadata.get('concert')
    if concert:
        concert = concert[0].get('title')
        
    title = metadata.get('title')
    length = metadata.get('length')
    
    raaga_inf = svara_freq.get('raaga')
    if raaga_inf:
        arohana = raaga_inf['arohana']
        avorahana = raaga_inf['avorahana']
    else:
        arohana = 'None provided'
        avorahana = 'None provided'
        
    return raaga, taala, album_artists, concert, title, length, arohana, avorahana