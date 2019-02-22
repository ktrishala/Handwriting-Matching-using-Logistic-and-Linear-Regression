######Script to be executed#######
import dataset_processing      #creating the four types of dataset files
while True:
    print('#####Select the Dataset as below:#####\n')
    print('Enter 1 for Human Observed Dataset with feature concatentation')
    print('Enter 2 for Human Observed Dataset with feature subtraction')
    print('Enter 3 for GSC Dataset with feature concatentation')
    print('Enter 4 for GSC Dataset with feature subtraction')
    dataset = input("Enter your choice: ")

    if(dataset=='1'):
        print('#####Select the Machine Learning Algorithm that you want to run:#####\n')
        print('Enter 1 for Linear Regression')
        print('Enter 2 for Logistic Regression')
        print('Enter 3 for Neural Network')
        model = input("Enter your choice: ")
        if(model=='1'):
            print('Running LinearRegression on HumanObservedconcDataset')
            import LinearReg_HumanObservedconcDataset

        elif(model=='2'):
            print('Running Logistic Regression on HumanObservedconcDataset')
            import LogisticReg_HumanObservedconcDataset

        elif(model=='3'):
            print('Running Neural Network on HumanObservedConcDataset')
            import Neural_HumanObservedConcDataset

        else:
            print('Invalid Input')
            exit()
    elif(dataset=='2'):
        print('#####Select the Machine Learning Algorithm that you want to run:#####\n')
        print('Enter 1 for Linear Regression')
        print('Enter 2 for Logistic Regression')
        print('Enter 3 for Neural Network')
        model = input("Enter your choice: ")
        if(model=='1'):
            print('Running LinearRegression on HumanObservedSubDataset')
            import LinearReg_HumanObservedSubDataset

        elif(model=='2'):
            print('Running Logistic Regression on HumanObservedSubDataset')
            import LogisticReg_HumanObservedSubDataset

        elif(model=='3'):
            print('Running Neural Network on HumanObservedsubDataset')
            import Neural_HumanObservedsubDataset

        else:
            print('Invalid Input')
            exit()
    elif(dataset=='3'):
        print('#####Select the Machine Learning Algorithm that you want to run:#####\n')
        print('Enter 1 for Linear Regression')
        print('Enter 2 for Logistic Regression')
        print('Enter 3 for Neural Network')
        model = input("Enter your choice: ")
        if(model=='1'):
            print('Running LinearRegression on GSCconcDataset')
            import LinearReg_GSCconcDataset

        elif(model=='2'):
            print('Running Logistic Regression on GSCconcDataset')
            import LogisticReg_GSCconcDataset

        elif(model=='3'):
            print('Running Neural Network on GSCconcDataset')
            import Neural_GSCconcDataset

        else:
            print('Invalid Input')
            exit()
    elif(dataset=='4'):
        print('#####Select the Machine Learning Algorithm that you want to run:#####\n')
        print('Enter 1 for Linear Regression')
        print('Enter 2 for Logistic Regression')
        print('Enter 3 for Neural Network')
        model = input("Enter your choice: ")
        if(model=='1'):
            print('Running LinearRegression on GSCSubDataset')
            import LinearReg_GSCSubDataset

        elif(model=='2'):
            print('Running Logistic Regression on GSCSubDataset')
            import LogisticReg_GSCSubDataset

        elif(model=='3'):
            print('Running Neural Network on GSCsubDataset')
            import Neural_GSCsubDataset

        else:
            print('Invalid Input')
            exit()

    res=input('Do you want to continue: [y/n]')
    if(res=='n'):
        exit()
