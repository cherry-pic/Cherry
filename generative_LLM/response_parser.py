


def parse_response(text,gold):
    text = text.lower()
    if "yes" in text:
        return 1
    elif "no" in text:
        return 0
    else:
        if gold==1:    #if hallucinates incorrect response, count it as error
            return 0
        else:
            return 1
