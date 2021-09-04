// Positions the user on the map
const drawUserPosition = (userLogs) => {
  const userPositionGroup = d3
    .select("#dataviz svg")
    .append("g")
    .attr("class", "userPositionGroup")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  userPositionGroup
    .selectAll(".userPositionIcon")
    .data(userLogs)
    .enter()
    .append("image")
    .attr("class", "userPositionIcon")
    // .attr("x", (d) => x(d.position.x))
    // .attr("y", (d) => y(d.position.y))
    .attr("x", (d) => newX(d.position.x))
    .attr("y", (d) => newY(d.position.y))
    .attr("xlink:href", (d) =>
      d.isCurrentUser ?
      "/images/map_images/current-user.png" :
      "/images/map_images/other-user.png"
    )
    .attr("width", 25)
    .attr("height", 25)
    .attr("r", 25)
    .attr("clip-path", "url(#clip)")
    .style("opacity", (d) => (d.isCurrentUser ? "1.0" : "0.25"))
    .on("mouseover", function (d) {
      Tooltip.style("opacity", 1)
        .style("left", d3.mouse(this)[0] + 60 + "px")
        .style("top", d3.mouse(this)[1] + 50 + "px")
        .style("pointer-events", "none")
        .html(
          d.isCurrentUser ?
          "You are here!" :
          `${d.username.toUpperCase()} is here!`
        );
    })
    .on("mouseleave", function (d) {
      Tooltip.style("opacity", 0);
    });
};

// Computes the average position of the user based on 3 recent clicks
const computeUserPosition = (userLogs) => {
  for (let iter = 0; iter < userLogs.length; iter++) {
    const position = {
      x: 0,
      y: 0
    };
    userLogs[iter]["logs"].forEach((log) => {
      position["x"] += resourceData[log["learningMapResourceId"]]["normX"];
      position["y"] += resourceData[log["learningMapResourceId"]]["normY"];
    });
    position["x"] /= 3;
    position["y"] /= 3;
    userLogs[iter]["position"] = position;
  }
  return userLogs;
};

// Fetches the list of users and then obtains their most recent click logs
// NOTE : Currently only the click log of a single user is being fetched
const fetchUserPosition = () => {
  // Clear the map
  d3.select(".userPositionGroup").remove();
  $.ajax({
    url: "/log/getAllUsersRecentLog",
    type: "get",
    contentType: "application/json",
    success: (response) => {
      let userLogs = response;
      userLogs = computeUserPosition(userLogs);
      drawUserPosition(userLogs);
    },
    error: (error) => {
      console.log(error);
    },
  });
};

fetchUserPosition();
setInterval(fetchUserPosition, 300000);