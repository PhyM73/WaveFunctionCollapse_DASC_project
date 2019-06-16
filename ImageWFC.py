import wfc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog

def image2matrix(image_path):
    """Convert image at `image_path` to matrix."""
    assert image_path[-3:] == 'png'
    im = matplotlib.image.imread(image_path)
    img = tuple(tuple(tuple(im[x][y]) for y in range(im.shape[1])) for x in range(im.shape[0]))
    return img


def mean_pixel(wave, position, i, j):
    """Get the weighted mean of the state space of position as the pixel there"""
    keys, values = list(wave[position].space.keys()), np.array(list(wave[position].space.values()))
    return tuple(
        map(lambda x: np.average(np.array(x), weights=values), zip(*(wave.patterns[index][i][j] for index in keys))))


def ImageProcessor(image_path, size, N, options):

    entry = image2matrix(image_path)

    def update(matrix, position, w, N):
        limit_i = N if position[0] == w.size[0] - 1 else 1
        limit_j = N if position[1] == w.size[1] - 1 else 1
        for i in range(limit_i):
            for j in range(limit_j):
                matrix[position[0] + i, position[1] + j] = mean_pixel(w, position, i, j)
        return matrix

    w = wfc.WaveFunction(size, entry, N=N, **options)
    fig = plt.figure(figsize=(4*(size[1]/size[0]), 4))
    matrix = np.array([[mean_pixel(w, (0, 0), 0, 0)] * size[1] for _ in range(size[0])])
    im = plt.imshow(matrix)

    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.pause(0.0001)

    for changed in w.observe(w.options['surv']):
        for pos in changed:
            matrix = update(matrix, pos, w, N)
            im.set_array(matrix)
        fig.canvas.draw()
        plt.pause(0.0000001)
    if tk.messagebox.askyesno(title='Save', message='Save the image?'):
        path = tkinter.filedialog.asksaveasfilename(defaultextension='.png', filetypes= [("PNG", ".png"), ("JPG" , ".jpg")],initialdir='result')
        fig.savefig(path, dpi=150, pad_inches = 0)
    plt.show()


root = tk.Tk()
root.title("WaveFunctionCollapse")
root.geometry("480x380")

frame1 = tk.Frame(root)
frame1.grid(row=0,column=0,padx=40,pady=30,sticky=tk.W)
frame2 = tk.Frame(root)
frame2.grid(row=1,column=0,padx=40,sticky=tk.W)

pms = [['N', tk.IntVar(),1,4,100,1],['width', tk.IntVar(),10,100,300,10],['height', tk.IntVar(),10,100,300,10]]
for i in range(len(pms)):
    tk.Label(frame1, text=pms[i][0]+':',width=8).grid(row=i,column=0,sticky=tk.NW)
    tk.Entry(frame1, textvariable=pms[i][1],width=3).grid(row=i,column=1, sticky=tk.NW)
    tk.Scale(frame1, variable =pms[i][1],from_=pms[i][2], to=pms[i][3], length=pms[i][4],tickinterval=pms[i][5],orient=tk.HORIZONTAL,showvalue=False).grid(row=i,column=2,sticky=tk.NW)


tk.Label(frame2, text='options:',width=8).grid(row=0, column=0, sticky=tk.W)
varis = [['All Rules', tk.IntVar()],['Surveil', tk.IntVar()],['Periodic Input', tk.IntVar()],
        ['Periodic Output', tk.IntVar()],['Rotation',tk.IntVar()],['Reflection', tk.IntVar()]]
varis[1][1].set(1)
for i, (key, value) in enumerate(varis):
    tk.Checkbutton(frame2, text=key, variable=value).grid(row=i//2, column=i%2+1,pady=5,sticky=tk.W)

path = tk.StringVar()
path.set('')

def get_image():
    path.set(tkinter.filedialog.askopenfilename(initialdir='samples'))
    return True

def main():
    options = {        
        'AllRules': varis[0][1].get(),
        'surveil': varis[1][1].get(),
        'PeriodicInput': varis[2][1].get(),
        'PeriodicOutput': varis[3][1].get(),
        'Rotation':varis[4][1].get(),
        'Reflection': varis[5][1].get()
    }
    try:
        ImageProcessor(path.get(), (pms[2][1].get(), pms[1][1].get()), N=pms[0][1].get(), options=options)
    except tk.TclError:
        pass
    except (AttributeError,OSError,AssertionError):
        tk.messagebox.showerror(title='OSError',message="Please open a PNG file!")
    except RuntimeError:
        tk.messagebox.showerror(title='CollapseError',message="Sorry for failed :(\nYou may try a smaller N or allow more rules!")
    except ValueError:
        tk.messagebox.showerror(title='SaveError',message="Only PNG or JPG files can be saved!!")

tk.Button(frame2,width=8, text="Open file", command=get_image).grid(row=3, column=1,pady=10,sticky=tk.W)
tk.Button(frame2,width=8, text="WFC !", command=main).grid(row=3, column=2,pady=10,sticky=tk.W)
tk.Button(frame2,width=8, text='Exit', command=root.quit()).grid(row=3, column=3,pady=10,sticky=tk.W)

tk.mainloop()