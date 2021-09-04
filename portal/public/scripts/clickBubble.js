const resourceData = {};
resData.forEach(
  (data) =>
    (resourceData[data._id] = {
      resourceId: data._id,
      normX: data.normX,
      normY: data.normY,
      resourceName: data.resourceId,
    })
);

// Draws the click bubble map using D3
const drawClickBubble = (resources) => {
  const clickBubbleGroup = d3
    .select("#dataviz svg")
    .insert("g", ":first-child")
    .attr("class", "clickBubbleGroup")
    .attr("pointer-events", "none")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Define a scale for radius of the bubble
  const radius = d3
    .scaleLinear()
    .domain(d3.extent(resources, (d) => d.count))
    .range([30, 100]);

  // Define a scale for the radius of the bubble
  const color = d3
    .scaleLinear()
    .domain(d3.extent(resources, (d) => d.count))
    .range(["paleturquoise", "darkblue"]);

  clickBubbleGroup
    .selectAll(".clickBubble")
    .data(resources)
    .enter()
    .append("circle")
    .attr("class", "clickBubble")
    // .attr("cx", (d) => x(d.normX))
    // .attr("cy", (d) => y(d.normY))
    .attr("cx", (d) => newX(d.normX))
    .attr("cy", (d) => newY(d.normY))
    .attr("r", (d) => radius(d.count))
    .style("fill", (d) => color(d.count))
    // .style("opacity", (d) => opacity(d.count))
    .style("opacity", 0.5)
    .attr("clip-path", "url(#clip)");
};

// Compute the number of clicks for each resource (that has been logged)
const countResourceClicks = (logs) => {
  // {resourceId : {resourceId, normX, normY, count}}
  let resources = {};

  logs.forEach((log) => {
    if (resources[log["learningMapResourceId"]] === undefined) {
      resources[log["learningMapResourceId"]] = {
        resourceId: log["learningMapResourceId"],
        normX: resourceData[log["learningMapResourceId"]]["normX"],
        normY: resourceData[log["learningMapResourceId"]]["normY"],
        count: 1,
      };
    } else {
      resources[log["learningMapResourceId"]]["count"] += 1;
    }
  });

  return Object.values(resources);
};

// Fetch the resource log -> compute resource click -> draw d3 map
const fetchResourcesLog = () => {
  // Clear the map
  d3.select(".clickBubbleGroup").remove();

  const resourceIds = [];

  resData.forEach((data) => {
    resourceIds.push(data._id);
  });

  $.ajax({
    url: "/log/getResourcesLog",
    type: "get",
    contentType: "application/json",
    data: {
      resourceIds: resourceIds,
    },
    success: (response) => {
      const logs = response;
      const resource = countResourceClicks(logs);
      drawClickBubble(resource);
    },
    error: (error) => {
      console.log(`ERROR on GET to /log/getResourcesLog!`);
      console.log(error);
    },
  });
};

fetchResourcesLog();

// Interval of refresh set for 5 minutes
setInterval(fetchResourcesLog, 300000);
