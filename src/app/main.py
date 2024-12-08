from azure.core.credentials import AzureKeyCredential
from azure.identity import ManagedIdentityCredential, ClientSecretCredential
from azure.storage.blob import BlobServiceClient, StorageStreamDownloader
from dotenv import load_dotenv
import os
import io
from typing import Union
from fastapi import FastAPI

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

endpoint = os.getenv("endpoint")
key = os.getenv("key")

cv_client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

app = FastAPI()
load_dotenv()
class Album():
    def __init__(self, id, title, artist, price, image_url):
         self.id = id
         self.title = title
         self.artist = artist
         self.price = price
         self.image_url = image_url

albums = [ 
    Album(1, "You, Me and an App Id", "Daprize", 10.99, "https://aka.ms/albums-daprlogo"),
    Album(2, "Seven Revision Army", "The Blue-Green Stripes", 13.99, "https://aka.ms/albums-containerappslogo"),
    Album(3, "Scale It Up", "KEDA Club", 13.99, "https://aka.ms/albums-kedalogo"),
    Album(4, "Lost in Translation", "MegaDNS", 12.99,"https://aka.ms/albums-envoylogo"),
    Album(5, "Lock Down Your Love", "V is for VNET", 12.99, "https://aka.ms/albums-vnetlogo"),
    Album(6, "Sweet Container O' Mine", "Guns N Probeses", 14.99, "https://aka.ms/albums-containerappslogo")
]

connection_str = os.getenv('AZURE_STORAGEBLOB_CONNECTIONSTRING')
blob_service_client = BlobServiceClient.from_connection_string(connection_str)


@app.get("/")
def read_root():
    return {"message": "Hello, World"}

@app.get("/albums")
def get_albums():
    return albums

@app.get("/env")
def get_env():
    # Zugriff auf die Variablen
    value = os.getenv('WERT3', 'default_value')
    #database_url = os.environ.get('VARIABLE_NAME')
    #api_key = os.environ.get('ANDERE_VARIABLE')
    return {f"Datenbank-URL: {value}"}

@app.get("/blob")
def get_blob():
    try:
        #endpoint = os.getenv("ai-ep")
        #key = os.getenv("ai-key")
        #print(os.environ.get('azure-storageblob-resourceendpoint-1e1f3'))
        #account_url = f"https://rgpeuwocr019118.blob.core.windows.net"  # os.environ.get("azure-storageblob-resourceendpoint-1e1f3")
        #cred = ManagedIdentityCredential()
        #blob_service_client = BlobServiceClient(account_url, credential=cred)
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGEBLOB_CONNECTIONSTRING'))

        print("\nListing blobs...")
        container_client = blob_service_client.get_container_client("processed")
        # List the blobs in the container
        blob_list_names = []
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            print("\t" + blob.name)
            blob_list_names.append(blob.name)
            ocr_image_file(blob.name)
        return blob_list_names
    except KeyError:
        print("Missing environment variable 'AI_SERVICE_ENDPOINT' or 'AI_SERVICE_KEY'")
        print("Set them before running this ocr.")
        exit()

@app.get("/ocrtest")
def get_ocrtest():
    return ocr_image_file("kl2809.jpeg")


@app.get("/ocr")
def get_ocr():
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGEBLOB_CONNECTIONSTRING'))

    # Instantiate a new ContainerClient
    #container_client = blob_service_client.get_container_client("processed")

    try:
        blob_client = blob_service_client.get_blob_client(container="processed", blob="kl2809.jpeg")
        with open(file=os.path.join(r'/code', blob_client.blob_name), mode="wb") as sample_blob:
            download_stream = blob_client.download_blob()
            sample_blob.write(download_stream.readall())

        stream = io.BytesIO()
        num_bytes = blob_client.download_blob().readinto(stream)
        print(f"Number of bytes: {num_bytes}")

        print(blob_client.blob_name)
        #return(blob_client.blob_name)
    except KeyError:
        print("Missing environment variable 'AI_SERVICE_ENDPOINT' or 'AI_SERVICE_KEY'")
        print("Set them before running this ocr.")
        exit()

    stream_down: StorageStreamDownloader = blob_client.download_blob()
    bytes_data = stream_down.readall()

    # Use Analyze image function to read text in image
    result = cv_client.analyze(
        image_data=bytes_data,
        visual_features=[VisualFeatures.READ]
    )
    # Print caption results to the console
    print("Image analysis results:")
    print(" Caption:")
    if result.caption is not None:
        print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

    # Display the image and overlay it with the extracted text
    if result.read is not None:
        print("\nText:")

        # Prepare image for drawing
        #image = Image.open(bytes_data)
        image = Image.open(io.BytesIO(bytes_data))

        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        for line in result.read.blocks[0].lines:
            print(f"  {line.text}")
            drawLinePolygon = True

            r = line.bounding_polygon
            bounding_polygon = ((r[0].x, r[0].y), (r[1].x, r[1].y), (r[2].x, r[2].y), (r[3].x, r[3].y))
            # Return the position bounding box around each line
            print("   Bounding Polygon: {}".format(bounding_polygon))
            if drawLinePolygon:
                draw.polygon(bounding_polygon, outline=color, width=1)

        # Save image
        plt.imshow(image)
        plt.tight_layout(pad=0)
        #outputfile = os.path.join('/code',
                                  #os.path.basename(blob_client.blob_name).split('.')[0] + '_processed' + '.jpg')
        #fig.savefig(outputfile)
        output_filename = blob_client.blob_name.split('.')[0] + '_processed.' + blob_client.blob_name.rsplit('.',1)[1]

        plt.title(output_filename)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        plt.close(fig)
        buf.seek(0)

        blob_client = blob_service_client.get_blob_client(container="processed", blob=output_filename)
        blob_client.upload_blob(buf, blob_type="BlockBlob", overwrite=True)
        buf.close()

        print('\n  Results saved in', output_filename)
        return(f'Upload der Datei: {output_filename} erfolgreich.')


def ocr_image_file(image_filename):
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGEBLOB_CONNECTIONSTRING'))

    # Instantiate a new ContainerClient
    #container_client = blob_service_client.get_container_client("processed")

    try:
        blob_client = blob_service_client.get_blob_client(container="processed", blob=image_filename)
        with open(file=os.path.join(r'/code', blob_client.blob_name), mode="wb") as sample_blob:
            download_stream = blob_client.download_blob()
            sample_blob.write(download_stream.readall())

        stream = io.BytesIO()
        num_bytes = blob_client.download_blob().readinto(stream)
        print(f"Number of bytes: {num_bytes}")

        print(blob_client.blob_name)
        #return(blob_client.blob_name)
    except KeyError:
        print("Missing environment variable 'AI_SERVICE_ENDPOINT' or 'AI_SERVICE_KEY'")
        print("Set them before running this ocr.")
        exit()

    stream_down: StorageStreamDownloader = blob_client.download_blob()
    bytes_data = stream_down.readall()

    # Use Analyze image function to read text in image
    result = cv_client.analyze(
        image_data=bytes_data,
        visual_features=[VisualFeatures.READ]
    )
    # Print caption results to the console
    print("Image analysis results:")
    print(" Caption:")
    if result.caption is not None:
        print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

    # Display the image and overlay it with the extracted text
    if result.read is not None:
        print("\nText:")

        # Prepare image for drawing
        #image = Image.open(bytes_data)
        image = Image.open(io.BytesIO(bytes_data))

        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        for line in result.read.blocks[0].lines:
            print(f"  {line.text}")
            drawLinePolygon = True

            r = line.bounding_polygon
            bounding_polygon = ((r[0].x, r[0].y), (r[1].x, r[1].y), (r[2].x, r[2].y), (r[3].x, r[3].y))
            # Return the position bounding box around each line
            print("   Bounding Polygon: {}".format(bounding_polygon))
            if drawLinePolygon:
                draw.polygon(bounding_polygon, outline=color, width=1)

        # Save image
        plt.imshow(image)
        plt.tight_layout(pad=0)
        #outputfile = os.path.join('/code',
                                  #os.path.basename(blob_client.blob_name).split('.')[0] + '_processed' + '.jpg')
        #fig.savefig(outputfile)
        output_filename = blob_client.blob_name.split('.')[0] + '_processed.' + blob_client.blob_name.rsplit('.',1)[1]

        plt.title(output_filename)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        plt.close(fig)
        buf.seek(0)

        blob_client = blob_service_client.get_blob_client(container="processed", blob=output_filename)
        blob_client.upload_blob(buf, blob_type="BlockBlob", overwrite=True)
        buf.close()

        print('\n  Results saved in', output_filename)
        return(f'Upload der Datei: {output_filename} erfolgreich.')

