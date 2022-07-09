sample['net_input']['src_tokens'][0]

Debuging in coref code at dockey: 'bn/voa/01/voa_0109_0'


Coref s2e
    - With batch:
        ('mention precision', 0.8929820950812924), ('mention recall', 0.8781623153207853), ('mention f1', 0.8855102040816326), ('precision', 0.8119616137410027), ('recall', 0.79486854814813), ('f1', 0.8033168271045601)

    - Without batch (single sample)
        ('mention precision', 0.8928902150426998), ('mention recall', 0.8781623153207853), ('mention f1', 0.8854650272945257), ('precision', 0.8118573582823948), ('recall', 0.79486854814813), ('f1', 0.8032657703367038)

Coref s2e list comprehension:
    - without comprehension:
        ('loss', 0.4487023401393819), ('precision', 0.8119616137410027), ('recall', 0.79486854814813), ('f1',0.8033168271045601)
    - with
