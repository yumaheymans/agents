from dotenv import load_dotenv
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPERDEV_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
OPENAI_MODEL_NAME='gpt-4o'

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serperdev_key = os.getenv("SERPERDEV_KEY")

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai_tools import tool
from openai import OpenAI
from langchain_openai import ChatOpenAI
from crewai_tools.tools import FileReadTool
import os, requests, re, mdpdf, subprocess

search_tool = SerperDevTool()

@tool
def generateimage(topic: str) -> str:
    """
    Generates an image for a given {topic}.
    Using the OpenAI image generation API,
    saves it in the current folder, and returns the image path.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.images.generate(
        model="dall-e-3",
        prompt=f"Image is about: {topic}. Style: Illustration. Create an illustration incorporating a vivid palette with an emphasis on shades of azure and emerald, augmented by splashes of gold for contrast and visual interest. The style should evoke the intricate detail and whimsy of early 20th-century storybook illustrations, blending realism with fantastical elements to create a sense of wonder and enchantment. The composition should be rich in texture, with a soft, luminous lighting that enhances the magical atmosphere. Attention to the interplay of light and shadow will add depth and dimensionality, inviting the viewer to delve into the scene. DON'T include ANY text in this image. DON'T include colour palettes in this image.",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    words = topic.split()[:5] 
    safe_words = [re.sub(r'[^a-zA-Z0-9_]', '', word) for word in words]  
    filename = "_".join(safe_words).lower() + ".png"
    filepath = os.path.join(os.getcwd(), filename)

    # Download the image from the URL
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        with open(filepath, 'wb') as file:
            file.write(image_response.content)
    else:
        print("Failed to download the image.")
        return ""

    return filepath

@tool
def convermarkdowntopdf(markdownfile_name: str) -> str:
    """
    Converts a Markdown file to a PDF document using the mdpdf command line application.

    Args:
        markdownfile_name (str): Path to the input Markdown file.

    Returns:
        str: Path to the generated PDF file.
    """
    output_file = os.path.splitext(markdownfile_name)[0] + '.pdf'
    
    # Command to convert markdown to PDF using mdpdf
    cmd = ['mdpdf', '--output', output_file, markdownfile_name]
    
    # Execute the command
    subprocess.run(cmd, check=True)
    
    return output_file

# Agents
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "Driven by curiosity, you're at the forefront of"
    "innovation, eager to explore and share knowledge that could change"
    "the world."
  ),
  max_iter=3,
  tools=[search_tool],
  allow_delegation=False
)

writer = Agent(
  role='Writer',
  goal='Narrate compelling and very detailed and well broken down stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "You write engaging blog posts about {topic}. Your writing should be very clearly structured with the right use of headers. You write very extensive and detailed blogs. You do this based on the research outcomes of the research_agent"
  ),
  max_iter=3,
  allow_delegation=False
)
from crewai import Task

image_generator = Agent(
  role='Image Generator',
  goal='Generate one image per chapter content provided by the story outliner. Start with Chapter number, chapter content, character details, detailed location information and detailed items in the location where the activity happens. Generate totally 5 images one by one. Final output should contain all the 5 images in json format.',
  backstory="A creative AI specialized in visual storytelling, bringing each chapter to life through imaginative imagery.",
  verbose=True,
  tools=[generateimage],
  allow_delegation=False
)

markdown_to_pdf_creator = Agent(
    role='PDF Converter',
    goal='Convert the Markdown file to a PDF document. story.md is the markdown file name.',
    backstory='An efficient converter that transforms Markdown files into professionally formatted PDF documents.',
    verbose=True,
    tools=[convermarkdowntopdf],
    allow_delegation=False
)

manager = Agent(
  role='Marketing Manager',
  goal='Responsible for a good outcome of for the request and selects the right agents to do the job',
  verbose=True,
  memory=True,
  backstory=(
    "You're the manager of the marketing team, you select the right agents at the right time to do the job and you review the work that you get back from the agents"
  ),
  allow_delegation=True
)

# Research task
research_task = Task(
  description=(
    "Identify the next big trend in {topic}."
    "Focus on identifying pros and cons and the overall narrative."
    "Your final report should clearly articulate the key points,"
    "its market opportunities, and potential risks."
  ),
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  tools=[search_tool],
  agent=researcher,
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging, and positive."
  ),
  expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
  tools=[search_tool],
  agent=writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)

task_image_generate = Task(
    description='Generate 5 images that captures the essence of the {topic}',
    agent=image_generator,
    expected_output='A digital image file that visually represents the overarching theme of the {topic}',
)

task_markdown_to_pdf = Task(
    description='Convert a Markdown file to a PDF document, ensuring the preservation of formatting, structure, and embedded images using the mdpdf library.',
    agent=markdown_to_pdf_creator,
    expected_output='A PDF file generated from the Markdown input, accurately reflecting the content with proper formatting. The PDF should be ready for sharing or printing.'
)

from crewai import Crew, Process

# Forming the crew with some enhanced configurations
crew = Crew(
  agents=[researcher, writer, image_generator, markdown_to_pdf_creator],
  tasks=[research_task, write_task, task_image_generate, task_markdown_to_pdf],
  process=Process.hierarchical, 
  memory=True,
  cache=True,
  max_rpm=100,
  manager_agent=manager
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
print(result)

