class Cat:
    def __init__(self, name = "noname", sex="nosex"): # constructor
        self.name = name
        self.sex = sex

    def speak(self):
        print(f"{self.name} {self.sex} says Meow!")
    

if __name__ == "__main__":
    belle = Cat(name="Belle", sex="F")
    belle.speak()
    
