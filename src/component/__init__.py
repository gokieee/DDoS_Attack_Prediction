inp = int(input("Enter a number"))

def odd_or_even(inp:int):
    
    if inp % 2 == 0:
        return "even"
    else:
        return "odd"
    

result= odd_or_even(inp)    
print(result)