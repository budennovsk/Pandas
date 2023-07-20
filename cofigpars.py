import configparser


config = configparser.ConfigParser()

config.read('settings.ini')
password = config['Login']['password']
font =config['Settings']['font']

print(password)
print(font)
