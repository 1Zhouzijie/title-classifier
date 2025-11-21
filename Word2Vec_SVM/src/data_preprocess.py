#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗与分词函数
"""

def simple_tokenize(text):
    """
    简单分词函数，将文本转成小写并按空格拆分
    Args:
        text (str): 原始文本
    Returns:
        list[str]: 分词列表
    """
    return str(text).lower().split()
