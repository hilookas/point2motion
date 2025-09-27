import cv2
import json
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from gradio_client import Client, handle_file

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_waypoints(image_rgb, task_instruction="Locate the green object."):
    prompt = f"You are currently a robot performing robotic manipulation tasks. The task instruction is: {task_instruction}. Use 2D points to mark the manipulated object-centric waypoints to guide the robot to successfully complete the task. You must provide the points in the order of the trajectory, and the number of points must be 8.\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. The answer consists only of several coordinate points, with the overall format being: <think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ..., [x8, y8]]</point></answer>"
    print("prompt: ============")
    print(prompt)
    print("====================")
    
    cv2.imwrite("upload.png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    client = Client("http://127.0.0.1:7860/")
    
    client.predict(
        file=handle_file('upload.png'),
        api_name="/add_file"
    )
    client.predict(
        text=prompt,
        api_name="/add_text"
    )
    result = client.predict(
        _chatbot=[[{"file":handle_file('upload.png'),"alt_text":None},None],[prompt,None]],
        api_name="/predict"
    )
    text_result = result[-1][-1]
    print("text_result: =======")
    print(text_result)
    print("====================")

    match = re.search(r"<[Aa]nswer>.*?<[Pp]oint>(.*?)</[Pp]oint>.*?</[Aa]nswer>", text_result, re.DOTALL)
    assert match, "No answer found"
    answer = match.group(1)
    points = np.array(json.loads(answer), dtype=np.float32)

    pick_point = points[0]
    place_point = points[-1]

    return pick_point, place_point, points, text_result

if __name__ == "__main__":
    image = cv2.imread("log_er1/20250923_215317_Pick up sponge and place it outside plate..png")
    image = cv2.resize(image, (640, 480))
    center_point, center_point_place, points, text_result = get_waypoints(image, "Pick up sponge and place it outside plate.png", "place")
    
    for point1, point2 in zip(points[:-1], points[1:]):
        cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 2)
    
    cv2.imwrite("image_with_points.png", image)
    
    # # plot points
    # for point in points:
    #     cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    # cv2.circle(image, (int(center_point[0]), int(center_point[1])), 5, (0, 255, 0), -1)
    # cv2.imwrite("image_with_points.png", image)
    
    # print(points)
    # print(center_point)