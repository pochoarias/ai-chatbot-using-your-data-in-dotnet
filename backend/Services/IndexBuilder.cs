using Pinecone;
using Microsoft.Extensions.AI;
using System.Collections.Immutable;

namespace ChatBot.Services;

public class IndexBuilder(
    StringEmbeddingGenerator embeddingGenerator,
    IndexClient pineconeIndex,
    WikipediaClient wikipediaClient,
    DocumentStore documentStore,
    DocumentChunkStore chunkStore,
    ArticleSplitter splitter)
{
    public async Task BuildIndexChunks(string[] pageTitles)
    {
        foreach (var title in pageTitles)
        {
            // Swap out the wikipediaClient here to connect to your own data source!
            var page = await wikipediaClient.GetWikipediaPageForTitle(title, full: true);
            var sections = wikipediaClient.SplitIntoSections(page.Content);

            var chunks = sections.SelectMany(section =>
                        splitter.Chunk(page.Title, section.Content, page.PageUrl, section.Title))
                        .Take(25) // limits the number of chunks per article to avoid creating too many vectors for this demo, remove or increase for better retrieval results at the cost of more vectors in your Pinecone index
                        .ToImmutableList();

            // Context for embedding generation, helps LLM understand the structure of the content and can improve retrieval relevance
            var stringsToEmbed = chunks.Select(c => $"{c.Title} > {c.Section}\n\n{c.Content}");

            // Makes a call to OpenAI to create an embedding from these strings
            var embeddings = await embeddingGenerator.GenerateAsync(
                stringsToEmbed,
                new EmbeddingGenerationOptions { Dimensions = 512 }
            );

            var vectors = chunks.Select((chunk, index) => new Vector
            {
                Id = chunk.Id,
                Values = embeddings[index].Vector.ToArray(),
                Metadata = new Metadata
                {
                    { "title", chunk.Title },
                    { "section", chunk.Section },
                    { "chunk_index", chunk.ChunkIndex }
                }
            });

            await pineconeIndex.UpsertAsync(new UpsertRequest
            {
                Vectors = vectors
            });

            foreach (var chunk in chunks)
            {
                chunkStore.SaveDocumentChunk(chunk);
            }

            // If you have rate limit issues with Pinecone (may happen based on your plan) then uncomment this Task.Delay()
            // see https://docs.pinecone.io/reference/api/database-limits#rate-limits
            await Task.Delay(500);
        }
    }

    public async Task BuildIndex(string[] pageTitles)
    {
        foreach (var title in pageTitles)
        {
            // Swap out the wikipediaClient here to connect to your own data source!
            var wikipediaPage = await wikipediaClient.GetWikipediaPageForTitle(title, full: false);

            
            // Makes a call to OpenAI to create an embedding from these strings
            var embeddings = await embeddingGenerator.GenerateAsync(
                [wikipediaPage.Content],
                new EmbeddingGenerationOptions
                {
                    Dimensions = 512
                }
            );

            var vectorArray = embeddings[0].Vector.ToArray();
            var pineconeVector =  new Vector
            {
                Id = wikipediaPage.Id,
                Values = vectorArray,
                Metadata = new Metadata
                {
                    { "title", wikipediaPage.Title },
                }
            };

            await pineconeIndex.UpsertAsync(new UpsertRequest
            {
                Vectors = [pineconeVector]
            });

            documentStore.SaveDocument(wikipediaPage);

            // If you have rate limit issues with Pinecone (may happen based on your plan) then uncomment this Task.Delay()
            // see https://docs.pinecone.io/reference/api/database-limits#rate-limits
            await Task.Delay(500);
        }
    }
}
