# PyQR
A simple QR Code generator and decoder written in Python 3.

## Dependency
+ Numpy
+ PIL
+ Matplotlib
+ array2gif
+ numba

## Usage
`from PyQR import QRCodeEncoder, QRCodeDecoder`
### Generate QR Code
+ `QRCodeEncoder().encode(message:str, ec_level:str, path:str)`
### Decode QR Code
+ `QRCodeDecoder().decode(path:str)`

## Note
+ This program was written two years ago, and I didn't expect to open source it at that time. That's why the code is a little bit messy. I'll try to refactor and reorganize it when I'm free.
+ You can read my articles about QR Code generation: https://yeecy.medium.com/%E5%A6%82%E4%BD%95%E8%A3%BD%E4%BD%9C-qr-code-0-%E5%89%8D%E8%A8%80-e464466dc321
+ Also, please note that `/Codecs/RSCodec.py` is released by someone at [Wikiversity](https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders). Since I can't find it's license, so I assume it's licensed under CC BY-SA 3.0. (Wikiversity says: "Text is available under the Creative Commons Attribution-ShareAlike License".)
