## Create cert.pem and key.pem for https (need to have openssl installed):
``openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes``