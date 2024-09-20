import os

# Define the rabbit animation frames
rabbit_frames = [
    """
     (\(\    
    ( -.-)    
    o_(")(")  --> Cooking, food almost done!
    """,
    """
     (\(\    
    ( o.o)    
    o_(")(")  --> Cooking, food almost done!
    """,
    """
     (\(\    
    ( ^_^)    
    o_(")(")  --> Smiling, food almost done!
    """
]

# Final frame after the custom code is done
finished_frame = """
     (\(\    
    ( ^_^)    
    o_(")_(")  --> Finished cooking!
"""

# Function to clear the screen
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

# Flag to indicate when `my_code` has completed
# Function to show the animation