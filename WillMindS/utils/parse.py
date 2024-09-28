
def unicode_strings_to_utf_8_strings(unicode_string):
    def bytes_to_unicode():
        """
        Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
        characters the bpe code barfs on.

        The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
        if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
        decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
        tables between utf-8 bytes and unicode strings.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    btu = bytes_to_unicode()
    utb = {value: key for key, value in btu.items()}
    # print(btu)
    # print(utb)
    utf8_10 = [utb[i] for i in unicode_string]
    # print(utf8_10)
    utf8_bytes = b''.join([b.to_bytes(1, byteorder='big') for b in utf8_10])
    # print(utf8_bytes)
    utf_8_string = utf8_bytes.decode('utf-8')
    # print(utf_8_string)
    return utf_8_string