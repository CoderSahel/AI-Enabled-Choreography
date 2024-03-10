import numpy as np
import pandas as pd
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib import animation
import cv2
from PIL import Image
import io
from sklearn.preprocessing import MinMaxScaler


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.axis('off')
limit=np.load("limits.npy")

def animate(shaped_data):
    ax.clear()
    #limit=np.load("limits.npy")

    # Set limits
    ax.set_xlim([limit[0], limit[1]])
    ax.set_ylim([limit[2], limit[3]])
    ax.set_zlim([limit[5], limit[4]])
        
    # Remove grid and ticks
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    
    # Plot each landmark
    for landmarks in shaped_data:
        ax.scatter(*landmarks, c='b', marker='.')
        
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    return img_array



def visualize(timestep, joints_data, starting_frame=0, scaled=False, save_name="output.avi"):
    plt.axis('off')  # Turn off the axis of the plot
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec using VideoWriter_fourcc
    result = cv2.VideoWriter(save_name, fourcc, 25.0, (480, 360))  # Create a VideoWriter object

    # If the data needs to be scaled
    if scaled:
        arr_reshaped = joints_data.reshape(-1, 1)  # Reshape the data to 2D for the scaler
        scaler = MinMaxScaler()  # Initialize a MinMaxScaler
        normalized_arr_reshaped = scaler.fit_transform(arr_reshaped)  # Fit and transform the data
        joints_data = normalized_arr_reshaped.reshape(joints_data.shape)  # Reshape the data back to original shape

    # Loop over each frame from starting_frame to timestep+starting_frame
    for i in range (starting_frame,timestep+starting_frame):
        print("frame:",i)  # Print the current frame number
        shaped_data = joints_data[:, i, :]  # Get the data for the current frame
        frame = animate(shaped_data)  # Fetch the current frame
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # Convert the color of the frame from RGB to BGR
        frame = frame[60:420, 80:560]  # Crop the frame
        result.write(frame)  # Write the frame to the file
        cv2.imshow('Frame', frame)  # Display the frame

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    result.release()  # Release the VideoWriter
    cv2.destroyAllWindows()  # Destroy all windows created by cv2
