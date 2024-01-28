''' 
Name_Validation Module
NameValidation is a module which is supposed to fatch images from good_image directory and move into bad_directories.
If image availbele in good_image directory satisfies all the validation condition then that image will be there in the good_directory else move into bad directory.
The ground truth of validation conditions should be fetched from source of truth called sot.json

Input : Fatch images from good_image directory
Output : Segregation of image in good_images and bad_iamges

Date of Development : 2-6-2021
Number of Revison : 1

'''

# Module Decleartion
# -------------------#
import json
import os
import sys
import shutil
from os import listdir
from os import path

# ---------------------#


class Name_Validation():
    def __init__(self,path):
        try:
            '''
            This is initialization function. This function is constructor.
            All the necessay variables are assigned in this function

            Input : path
            Output : All the necessary variable asignment

            '''
            # Initialize the directory path variable
            self.path = path

            # Checking availability of good_image directory
            
            if not os.path.isdir(self.path + 'good_image'):
                print("good_image directory  not found ")
                raise(" Good image directory not found")
            else:
                print("Good_Image Directory  found")

            # Load JSON source of truth(SOT) file
            self.sot=json.load(open(self.path+'source_of_truth/sot.json',"r"))
            print("SOT loaded successfully")

            print("Name Validation")
            print("Name Validation method Intialized")
        except Exception as e:
            print("Error in Name Validation Intialtion")
            print(f"Error on line number{sys.exc_info()[-1].tb_lineno}")
            print(str(e))
    
    def check_directory(self):
        try:
            
            '''
            This method create directory for the bad_image. If the directory exists 
            then it will be delted and create a new directory
            
            Input : N/A
            Output= If bad_drectory not exists then bad_directory is created 
                    else it will delte previous directory and create new one

            '''    
            # directory_name  to be created
            folder_name = "bad_image"

            # destination path directory to be created
            self.folder_path=self.path+folder_name
            
            # check if directory is present or not  
            CHECK_FOLDER = path.isdir(self.folder_path)

            # If folder doesn't exist, then create it.
            # else if exist delete the directory then create new directory
            if  not CHECK_FOLDER:
                print(f"{folder_name} directory  not present")
                print(f"Creating {folder_name} directory")
                os.makedirs(self.folder_path)
                print(f"{folder_name} directory create successfully",end='\n')
        
            else:
                print(f"{folder_name} directory already exists.")
                print(f"Deleting old {folder_name} directory")
                shutil.rmtree(self.folder_path)
                print(f"Creating {folder_name} directory")
                os.makedirs(self.folder_path)
                print(f"{folder_name} directory create successfully",end='\n')

        except Exception as e:
            print("Directory Creation error")
            print(f"Error on line number{sys.exc_info()[-1].tb_lineno}")
            print(str(e))

    def person_name_validation(self):
        try:
            '''
                This method is responsible for validation and moving  image form good_image directory
                to bad_image directory if validation condition doesn't satisfies.

                Input: N/A
                Output: Separate all the images to respective directory   
               
            '''
            print("Person Name validation method Checking")

            # listing all .jpg file name into self.jpg_file
            self.jpg_file=[file_name for file_name in listdir(self.path+'good_image') if file_name.split('.')[-1]==self.sot['format']]

            # finding the length of file name like (first_name and last_name) then its length is 2
            self.file_name_format_len=len(self.sot['name'].split(' '))

            # check if directory containing image or not
            if len(self.jpg_file): 
                

                for file_name in self.jpg_file:
                    
                    # Extract name form file_name 
                    self.name=file_name.split('.')[0]

                    # split the name by spaces
                    self.split_name=self.name.split(' ')

                    # source file path
                    self.source_path=self.path+'good_image/'+file_name

                    if len(self.split_name)!=self.file_name_format_len or (self.split_name[0]=='' or self.split_name[1]==''):
                        
                        # destination location path for bad_image
                        self.move_to=self.path+'bad_image'

                        # Moving the file bad_directory
                        shutil.move(self.source_path,self.move_to)
                        print("Moving to bad_image Directory")
            else:
                print("Directory does not contain any image")
        
        except Exception as e:
            print("Person Name Validation method error")
            print(f"Error on line number{sys.exc_info()[-1].tb_lineno}")
            print(str(e))
        
    def package(self):
        try:
            self.check_directory()
            self.person_name_validation()
            print("Name validation Complete")
        except Exception as e:
            print("Name validation error in package")
            print(f"Error on line number{sys.exc_info()[-1].tb_lineno}")
            print(str(e))

