import jieba
import os

cache_path = '/var/folders/_9/63sqp4sd22sfwk505637vvsh0000gn/T/jieba.cache'


def _not_empty(s: str):
    return s and s.strip()


def jdwords(s: str):
    s = s.replace('\t', '\n').split('\n')
    s = filter(_not_empty, s)
    cut = [jieba.lcut(x) for x in s]
    return cut


if os.path.exists(cache_path):
    os.remove(cache_path)
