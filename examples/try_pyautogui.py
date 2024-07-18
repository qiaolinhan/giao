import pyautogui

# Get the current screen size
screenWidth, screenHeight = pyautogui.size()
print("======>[INFO] My screen width and height:", screenWidth, "x", screenHeight)

# Get the current mouse pointing position on the screen
currentMouseX, currentMouseY = pyautogui.position()
print("======>[INFO] My mouse is now at the position of:", currentMouseX, currentMouseY)

# Move the mouse to target postion and click
targetPosition = (581, 249)
pyautogui.click(targetPosition)
pyautogui.click(targetPosition)

