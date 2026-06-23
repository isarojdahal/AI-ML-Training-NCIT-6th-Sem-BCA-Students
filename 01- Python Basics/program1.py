grade_point = float(input("Enter your Grade point : "))

if grade_point >4 or grade_point<0:
    print("Invalid Grade")
    exit()
    
if grade_point >= 3.50:
    print("Grade : A+")
elif grade_point >= 3.00:
    print("Grade : B")
elif grade_point >= 2.00:
    print("Grade : C")

else:
    print("Grade : D")