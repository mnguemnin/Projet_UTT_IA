from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the search tool
search = DuckDuckGoSearchRun()


class ImageSearchTool:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()

    def search_images(self, query):
        # Search specifically for images using a more refined query
        query_with_images = f"{query} images"
        results = self.search.run(query_with_images)

        # Debugging: print the results to inspect the structure
        print("Search Results:", results)

        # If results are a string, it could be a plain text result, not JSON or dict.
        if isinstance(results, str):
            return "No image data found or results are in an unexpected format."

        # Extract image URLs from the search results (assuming it's a dict)
        image_urls = []
        for result in results.get("results", []):
            if 'image' in result:  # Check if there's an 'image' key
                image_urls.append(result['image'])
        return image_urls


# Create instance and search for images related to 'cat'
image_search_tool = ImageSearchTool()
print(image_search_tool.search_images('cat'))
