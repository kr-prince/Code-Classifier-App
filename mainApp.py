#!/usr/bin/env python
# coding: utf-8

from tkinter.ttk import Frame, Progressbar
from tkinter.filedialog import askopenfilename
from tkinter import Spinbox, Scrollbar, Button, Text, Label, font, messagebox, Toplevel
from tkinter import Tk, LEFT, RIGHT, TOP, BOTTOM, END, X,Y, BOTH, NONE, WORD

from CodeClassifier import CodeClassifier
from CodeClassifierTrain import rgxPattern, custom_tokenizer


class ClassifyApp:
    def __init__(self):
        self.root = Tk()
        self.code_classifier = None
    
        # Dividing the root into Frames(for control and code part)
        self.controlFrame = Frame(self.root, width=10)
        self.codeFrame = Frame(self.root)

        # Define custom fonts for button and code editor
        self.buttonfont = font.Font(family='System', size=12)
        self.codefont = font.Font(family='Lucida Console', size=10)

        # Define Controls - Buttons, Text and stuffs
        self.uploadbutton = Button(self.controlFrame, text='Upload',height=1, width=12,
                                      command=lambda: self.uploadFile())
        self.featureSpinner = Spinbox(self.controlFrame, from_=5, to=20, width=15)
        self.predictbutton = Button(self.controlFrame, text='Predict', fg='white', 
                        bg='dark green',height=2, width=12, command=lambda : self.predictLang())
        self.inputCode = Text(self.codeFrame, bg='white', wrap=WORD, spacing1=8)
        self.scrollBar = Scrollbar(self.inputCode)
        self.predictOut = Text(self.codeFrame, bg='light cyan', height=5, state='disabled', spacing1=5)
        
        
        # Define Label
        self.label = Label(self.controlFrame, text='No of Features')
        
        # Pack All the controls - Frames, Text, Button etc
        self.controlFrame.pack(side=LEFT, expand=False, fill=Y)
        self.codeFrame.pack(side=RIGHT, expand=True, fill=BOTH)
        self.uploadbutton['font'] = self.buttonfont
        self.uploadbutton.pack(expand=False, padx=10, pady=15)
        self.label.pack(anchor='w', expand=False, padx=10, pady=1)
        self.featureSpinner.pack(expand=False, padx=10, pady=1)
        self.predictbutton['font'] = self.buttonfont
        self.predictbutton.pack(side=BOTTOM, expand=False, padx=10, pady=15)
        self.inputCode['font'] = self.codefont
        self.inputCode.pack(fill=BOTH, expand=True)
        self.predictOut['font'] = self.codefont
        self.predictOut.pack(fill=X, expand=False)
        self.scrollBar.pack(side=RIGHT,fill=Y)
        # Scrollbar will adjust automatically according to the content         
        self.scrollBar.config(command=self.inputCode.yview)
        self.inputCode.config(yscrollcommand=self.scrollBar.set)
        self.root.title('Code Predict App')
    
    
    def validate(self):
        # This function validates all the control values
        status = True
        nfeatures = self.featureSpinner.get().strip()
        code = self.inputCode.get('1.0', END).strip()
        if not nfeatures.isdigit():
            status = 'No. of Features should be integer.'
        elif not 0 <= int(nfeatures) <= 20:
            status = 'No. of Features range should be (0-20).'
        elif not code:
            status = 'Code editor is blank.'
        elif self.code_classifier is None:
            status = 'Classifier not loaded.'
        else:
            status = True
        return status

    
    def uploadFile(self): 
        file = askopenfilename(defaultextension=".txt", 
                                      filetypes=[("All Files","*.*"), 
                                        ("Text Documents","*.txt")]) 
        if file == "": 
            # no file to open 
            file = None
        else:
            # Try reading the file
            try:
                code = ''
                with open(file, 'r', encoding='latin-1') as file:
                    code = file.read()
                
                # clear code area and re-write
                self.inputCode.delete(1.0,END) 
                self.inputCode.insert(1.0,code)
            except Exception as ex:
                messagebox.showerror("File Reading failed", str(ex))
    
    
    def predictLang(self): 
        try:
            status = self.validate()
            if status is not True:
                messagebox.showerror("Validation failed", status)
            else:
                self.inputCode.tag_remove('features', '1.0', END)
                code = self.inputCode.get('1.0', END)
                nfeatures = int(self.featureSpinner.get().strip())
                langs, confidence, top_features = self.code_classifier.classify(code, nfeatures)

                # Highligh the top features in the Code
                for feats in top_features:
                    idx = '1.0'
                    while True:
                        idx = self.inputCode.search(feats, idx, stopindex=END, nocase=0)
                        if not idx: break
                        lastidx = '% s+% dc' % (idx, len(feats)) 
                        self.inputCode.tag_add('features', idx, lastidx)
                        idx = lastidx
                self.inputCode.tag_config('features', foreground ='red', background='light yellow')
                self.inputCode.focus_set()

                # Populate the Language and its Confidence Score
                self.predictOut.configure(state='normal')
                self.predictOut.delete('1.0', END)
                for i in range(len(langs)):
                    self.predictOut.insert(END, '\n  '+str(i+1)+'. {:10}'.format(langs[i].title())+\
                                ' |'+'{:50}'.format('█'*(int(confidence[i])//2))+'|  '+\
                                str(confidence[i])+'%')
                self.predictOut.tag_add('bestGuess', '2.0', '2.end')
                self.predictOut.tag_config('bestGuess', foreground='red')
                self.predictOut.configure(state='disabled')
        except Exception as ex:
            messagebox.showerror("Application Error", str(ex))
    
    # def display_message(self, message):
    #     self.predictOut.configure(state='normal')
    #     self.predictOut.delete('1.0', END)
    #     self.predictOut.insert(END, str(message))
    #     self.predictOut.configure(state='disabled')

    
    def load_classifier(self):
        if self.code_classifier is None:
            self.code_classifier = CodeClassifier()
    
    
    def run(self):
        width = int(self.root.winfo_screenwidth()*0.8)
        height = int(self.root.winfo_screenheight()*0.8)
        left = (self.root.winfo_screenwidth() - width) // 2
        top = (self.root.winfo_screenheight() - height) // 3
        
        self.root.geometry('%dx%d+%d+%d'%(width, height, left, top))
        self.root.after(0, self.load_classifier)
        self.root.focus_force()
        self.root.mainloop()



if __name__ == "__main__":
    # Launch the main app
    app = ClassifyApp()
    app.run()

