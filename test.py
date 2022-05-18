values = [('Stop', 170.0), ('North', 170.0), ('South', 170.0)]
goldenMove = max(values, key=lambda pair: pair[1] and pair[0] != 'Stop')
print(goldenMove)