// This is part of data visualization by Swastik (GitHub: swax06)
var path_visible = true,
  plabel_visible = true,
  path_avaliable = true,
  all_visible = true;
var grx = 0,
  gry = 0,
  gsx = 999999,
  gsy = 999999;
var duration = 0,
  inView = 2,
  zoomLevel = 1,
  i = 6;
var pathTran = [],
  pathFilter = [],
  defaultWords = [],
  resTran = [];
var resFilter = [],
  words = [],
  mapCloud = [];
var cycle,
  timeout = null;
var placeholder =
  '<div class="card border-dark mb-6" style="max-width: 60rem; margin-top:10px; margin-bottom: 10px;">' +
  '              <div class="card-body">' +
  '                <div class="ui placeholder">' +
  '                    <div class="medium line"></div>' +
  '                    <div class="short line"></div>' +
  '                    <div class="short line"></div>' +
  "                </div>" +
  "                <br/><br/>" +
  '                <div class="ui placeholder">' +
  '                    <!-- <div class="short line"></div> -->' +
  '                    <div class="short line"></div>' +
  "                </div>" +
  "              </div> " +
  "            </div>";

if (pathData.length == 0) {
  path_avaliable = false;
  path_visible = false;
}

while (i--) {
  defaultWords.push([]);
}

for (var i = 0; i < pathData.length; i++) {
  val = JSON.parse(JSON.stringify(pathData[i]));
  for (var j = 0; j < pathData[i].values.length; j++) {
    val.values[j].normX = -1000;
    val.values[j].normY = -1000;
  }
  if (path_avaliable)
    defaultWords[pathData[i].values[0].pathwayLod - 1].push(pathData[i]);
  pathTran.push(val);
}

for (var i = 0; i < resData.length; i++) {
  gsx = Math.min(gsx, resData[i].normX);
  grx = Math.max(grx, resData[i].normX);
  gsy = Math.min(gsy, resData[i].normY);
  gry = Math.max(gry, resData[i].normY);
}

function draw(container) {
  $.ajax({
    url: "/resources/getSelectedResource",
    type: "get",
    contentType: "application/json",
    data: {
      resourceName: resourceNames
    },
    success: function (response) {
      if (container.id == "resourceCards") resourceList = response.resourceList;
      if (container.id == "resourceCards")
        markedResourceList = response.resourceList;
      username = response.username;
      user = response.user;
      $.ajax({
        url: "/resources/drawSelectedResource",
        type: "get",
        contentType: "application/json",
        data: {
          pathwayId: pathwayId
        },
        success: function (response) {
          container.innerHTML = response;
        },
        error: function (xhr) {
          //console.log("error fetching resources")
        },
      });
    },
    error: function (xhr) {
      //console.log("error sending resource data");
    },
  });
}

var resourceNames = [];
var pathwayId;
function listResources(resources) {
  container = document.getElementById("resourceCards");
  if (path_avaliable)
    {
//       document.getElementById("listTitle").innerText = resources[0].pathwayId;
      document.getElementById('listTitle').innerHTML = '<h5 id="listTitle" style="padding-top: 7px;">'+resources[0].pathwayId+'</h5><div class="btn-group"><a class="btn btn-round btn-outline-primary my-2 my-sm-0" href="/discussions/p/'+resources[0].pathwayId+'">Discussions</a><form action="/learningMaps/dataInLocalStorage" method="get"><button type="submit" id="create-trailer" class="btn btn-round btn-outline-primary my-2 my-sm-0" style="margin-left:3px;">Create Trailer</button></form></div>';
      pathwayId=resources[0].pathwayId;
    }
  else document.getElementById("listTitle").innerText = "";
  resourceNames = [];
  for (var i = 0; i < resources.length; i++) {
    // TODO : Don't use resourceId for resource name
    // resourceNames.push(resources[i].resourceId);
    resourceNames.push(resources[i].resourceName);
  }
  draw(container);

}

// set contains marked resources
var markedSet = new Set();

function toggleResource(resource) {
  if (markedSet.has(resource)) markedSet.delete(resource);
  else markedSet.add(resource);

  container = document.getElementById("markedResources");
  container.innerHTML = "";
  resourceNames = [];
  markedSet.forEach(function (resource) {
    // TODO : Don't use resourceId for resource name
    resourceNames.push(resource.resourceId);
  });
  if (resourceNames.length > 0) draw(container);
}

// mapping of resourcesIds to list of tags // used for drawing word cloud
var tagMap = {};
var mapTagMap = {};
for (var i = 0; i < topics.length; i++) {
  if (topics[i].resourceId in tagMap)
    tagMap[topics[i].resourceId].push({
      text: topics[i].topicName,
      size: 15 + Math.min(10000 * topics[i].resourceMappedProbability, 25),
    });
  else
    tagMap[topics[i].resourceId] = [{
      text: topics[i].topicName,
      size: 15 + Math.min(10000 * topics[i].resourceMappedProbability, 25),
    }, ];
  if (!(topics[i].topicName in mapTagMap))
    mapTagMap[topics[i].topicName] = {
      text: topics[i].topicName,
      size: 15 + Math.min(10000 * topics[i].resourceMappedProbability, 25),
    };
}

for (i in mapTagMap) {
  mapCloud.push(mapTagMap[i]);
}

// Set the dimensions and margins of the graph
let parentWidth = d3.select("#dataviz").style("width");
parentWidth = +parentWidth.substr(0, parentWidth.indexOf("p"));

let margin = {
  top: 30,
  right: 30,
  bottom: 30,
  left: 30
};
let width = parentWidth - margin.left - margin.right;
let height = 700 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3
  .select("#dataviz")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

//creating a tooltip
var Tooltip = d3
  .select("#dataviz")
  .append("div")
  .style("position", "absolute")
  .style("opacity", 0)
  .attr("class", "tooltip")
  .style("background-color", "#FFFFFF")
  .style("border", "1px solid #E0E1E2")
  .style("border-radius", "4px")
  .style("padding", "5px")
  .style("box-shadow", "1px 1px #A0A1A2");

// set the dimensions and margins of the wordcloud
var margin_cloud = {
    top: 10,
    right: 10,
    bottom: 10,
    left: 10
  },
  width_cloud = 575 - margin_cloud.left - margin_cloud.right,
  height_cloud = 600 - margin_cloud.top - margin_cloud.bottom;

// the word cloud's SVG element
var svg_cloud = d3v3
  .select("#word_cloud")
  .append("svg")
  .attr("width", width_cloud + margin_cloud.left + margin_cloud.right)
  .attr("height", height_cloud + margin_cloud.top + margin_cloud.bottom)
  .append("g")
  .attr(
    "transform",
    "translate(" +
    (width_cloud + margin_cloud.left + margin_cloud.right) / 2 +
    "," +
    (height_cloud + margin_cloud.top + margin_cloud.bottom) / 2 +
    ")"
  );

// A color scale: one color for each group
var myColor = d3.scaleOrdinal(d3.schemeTableau10); //https://github.com/d3/d3-scale-chromatic

function wordCloud() {
  var fill = d3.scaleOrdinal(d3.schemeDark2);
  //Draw the word cloud
  function draw(words) {
    var cloud = svg_cloud.selectAll(".cloudText").data(words, function (d) {
      return d.text;
    });

    //Entering words
    cloud
      .enter()
      .append("text")
      .style("font-family", "sans-serif")
      .attr("class", "cloudText")
      .style("fill", function (d, i) {
        return fill(i);
      })
      .attr("text-anchor", "middle")
      .attr("font-size", 1)
      .text(function (d) {
        return d.text;
      });

    //Entering and existing words
    cloud
      .transition()
      .duration(1000)
      .style("font-size", function (d) {
        return d.size + "px";
      })
      .attr("transform", function (d) {
        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
      })
      .style("fill-opacity", 1);

    //Exiting words
    cloud
      .exit()
      .transition()
      .duration(200)
      .style("fill-opacity", 1)
      .attr("font-size", 1)
      .remove();
  }

  //Use the module pattern to encapsulate the visualisation code. We'll
  // expose only the parts that need to be public.
  return {
    //Recompute the word cloud for a new set of words. This method will
    // asycnhronously call draw when the layout has been computed.
    //The outside world will need to call this function, so make it part
    // of the wordCloud return value.
    update: function (words) {
      d3.layout
        .cloud()
        .size([width_cloud - 20, height_cloud - 45])
        .words(words)
        .padding(8)
        .rotate(function () {
          return ~~(Math.random() * 2) * 90;
        })
        .fontSize(function (d) {
          return d.size | 0;
        })
        .on("end", draw)
        .start();
    },
  };
}

var myWordCloud = wordCloud();

// Add X Axis
var x = d3.scaleLinear().range([0, width]);
var xAxis = svg.append("g").attr("transform", "translate(0," + height + ")");

// Add Y axis
var y = d3.scaleLinear().range([height, 0]);
var yAxis = svg.append("g");

let resourceSmallImageSize = "25";
let resourceLargeImageSize = "50";
let clickCount = 0;

// Add a clipPath: everything out of this area won't be drawn.
var clip = svg
  .append("defs")
  .append("svg:clipPath")
  .attr("id", "clip")
  .append("svg:rect")
  .attr("width", width)
  .attr("height", height)
  .attr("transform", "translate(" + 0 + "," + 0 + ")");

var zoom = d3
  .zoom()
  .scaleExtent([0.95, 20]) // This control how much you can zoom-out (x0.95) and zoom-in (x20)
  .extent([
    [0, 0],
    [width, height],
  ])
  .on("zoom", updateChart);

// This add an invisible rect on top of the chart area. This rect can recover pointer events: necessary to understand when the user zoom
var gElem = svg.append("g").call(zoom);
gElem
  .append("rect")
  .attr("width", width)
  .attr("height", height)
  .style("fill", "none")
  .style("pointer-events", "all")
  .attr("transform", "translate(" + 0 + "," + 0 + ")");

function updateAxis(newX, newY, ti) {
  xAxis.transition().duration(ti).call(d3.axisBottom(newX));
  yAxis.transition().duration(ti).call(d3.axisLeft(newY));
}

var cloud_label = svg_cloud
  .append("text")
  .attr("dx", "-280")
  .attr("dy", "290")
  .style("font-size", "20")
  .style("pointer-events", "none");

var LOD_label = svg
  .append("text")
  .attr("dx", "850")
  .attr("dy", "5")
  .style("font-size", "15")
  .style("pointer-events", "none")
  .text("hi");

function getTags(d) {
  cloud_label
    .text(d.values[0].pathwayId)
    .style("fill", myColor(d.values[0].pathwayId));
  var tags = [];
  var temp = {};
  for (var i = 0; i < d.values.length; i++) {
    for (var j = 0; j < tagMap[d.values[i].resourceId].length; j++) {
      if (!(tagMap[d.values[i].resourceId][j].text in temp)) {
        temp[tagMap[d.values[i].resourceId][j].text] =
          tagMap[d.values[i].resourceId][j].size;
        tags.push({
          text: tagMap[d.values[i].resourceId][j].text,
          size: tagMap[d.values[i].resourceId][j].size,
        });
      }
    }
  }
  return tags;
}

var sel = d3
  .line()
  .curve(d3.curveStepAfter) //https://github.com/d3/d3-shape/blob/master/README.md#curves //curveNatural
  .x((d) => x(d.normX))
  .y((d) => y(d.normY));

var lines = svg.selectAll(".lineTest");
lines
  .data(pathData)
  .enter()
  .append("path")
  .attr("class", "lineTest")
  .attr("clip-path", "url(#clip)")
  .attr("fill", "none")
  .on("mouseover", function (d) {
    t = d.values[0].pathwayLod - 0.5;
    if (
      !all_visible ||
      zoomLevel > 6 ||
      (zoomLevel - 1 <= t && zoomLevel > t)
    ) {
      if (d3.select(this).attr("stroke-width") === "2")
        d3.select(this).attr("stroke-width", 3.5);
      if (path_visible) Tooltip.style("opacity", 1);
    }
  })
  .on("mousemove", function (d) {
    Tooltip.html("Path_ID: " + d.values[0].pathwayId)
      .style("left", d3.mouse(this)[0] + 60 + "px")
      .style("top", d3.mouse(this)[1] + 50 + "px")
      .style("pointer-events", "none");
  })
  .on("mouseleave", function (d) {
    // t = d.values[0].pathwayLod - 0.5; if(!all_visible || (zoomLevel > 6 || zoomLevel - 1 <= t && zoomLevel > t))
    if (d3.select(this).attr("stroke-width") === "3.5")
      d3.select(this).attr("stroke-width", 2);
    Tooltip.style("opacity", 0);
  })
  .on("click", function (d) {
    t = d.values[0].pathwayLod - 0.5;
    if (
      path_visible &&
      (!all_visible || zoomLevel > 6 || (zoomLevel - 1 <= t && zoomLevel > t))
    ) {
      var lines = svg.selectAll(".lineTest");
      lines.attr("stroke-width", 2);
      var dots = svg.selectAll(".myDots");
      //   dots.attr("r", 4);
      dots.attr("width", resourceSmallImageSize);
      dots.attr("height", resourceSmallImageSize);
      d3.select(this).attr("stroke-width", 4.5);
      clearInterval(cycle);
      myWordCloud.update(getTags(d));
      listResources(d.values);
    }
  })
  .attr("stroke", (d) => myColor(d.values[0].pathwayId))
  .attr("stroke-width", 2)
  .attr("d", (d) => sel(d.values));

// Three function that change the tooltip when user hover / move / leave a cell
var mouseover = function (d) {
  t = d.pathwayLod - 0.5;
  if (
    !path_visible ||
    !all_visible ||
    zoomLevel > 6 ||
    (zoomLevel - 1 <= t && zoomLevel > t)
  ) {
    Tooltip.style("opacity", 1);
    if (d3.select(this).attr("r") == resourceSmallImageSize) {
      d3.select(this)
        .attr("width", resourceLargeImageSize)
        .attr("height", resourceLargeImageSize);
    }
    // if (d3.select(this).attr("r") === "4") d3.select(this).attr("r", 6);
  }
};

var mousemove = function (d) {
  Tooltip.html(
      "Norm_X:           " + d.normX +
      "<br>Norm_Y:       " + d.normY +
      // TODO : Don't use resourceId
      "<br>Topic:        " + d.resourceName +
      "<br>Sequence no.: " + d.sequenceId
    )
    .style("left", d3.mouse(this)[0] + 60 + "px")
    .style("top", d3.mouse(this)[1] + 50 + "px")
    .style("pointer-events", "none");
};

var mouseleave = function (d) {
  Tooltip.style("opacity", 0);
  if (d3.select(this).attr("r") == resourceLargeImageSize) {
    d3.select(this)
      .attr("width", resourceSmallImageSize)
      .attr("height", resourceSmallImageSize);
  }
  //   if (d3.select(this).attr("r") === "6") d3.select(this).attr("r", 4);
};

var dots = svg.selectAll(".myDots");

dots
  .data(resData)
  .enter()
  //   .append("circle")
  .append("image")
  .attr("class", "myDots")
  .attr("xlink:href", (d) =>
    d.resourceType === "Pdf" ?
    "/images/map_images/pdf-file.png" :
    d.resourceType === "Video" ?
    "/images/map_images/youtube.png" :
    "/images/map_images/txt-file.png"
  )
  .attr("width", resourceSmallImageSize)
  .attr("height", resourceSmallImageSize)
  .attr("clip-path", "url(#clip)")
  .on("mouseover", mouseover)
  .on("mousemove", mousemove)
  .on("mouseleave", mouseleave)
  .on("click", function (d) {
    $.ajax({
      url: "/log/logResourceClick",
      type: "get",
      contentType: "application/json",
      data: {
        resource: d,
      },
      success: (response) => {
        clickCount++;
        if (clickCount == 3) {
          clickCount = 0;
          $("#refreshButton").transition("flash");
        }
      },
      error: (err) => {
        console.log(`ERROR on GET to /log/logResourceClick`);
        console.log(err);
      },
    });

    if (!isCloudSideBarOpen) {
      isCloudSideBarOpen = true;
      $("#dataviz").parent().attr("class", "ten wide column");
      $("#cloudSideBar").parent().attr("class", "six wide column");
      $("#cloudSideBar").show(100);
      $("#cloudSideBarButton i").attr("class", "angle double right icon");
      let newMapWidth = $("#dataviz").width();
      d3.select("#dataviz svg").attr("width", newMapWidth);
      clip.attr("width", newMapWidth);
      gElem.attr("width", newMapWidth);
      gElem.select("rect").attr("width", newMapWidth);
      x.range([0, newMapWidth]);
    }

    t = d.pathwayLod - 0.5;
    if (
      !path_visible ||
      !all_visible ||
      zoomLevel > 6 ||
      (zoomLevel - 1 <= t && zoomLevel > t)
    ) {
      clearTimeout(timeout);
      ths = d3.select(this);
      timeout = setTimeout(function () {
        clearInterval(cycle);
        var dots = svg.selectAll(".myDots");
        // dots.attr("r", 4);
        dots.attr("width", resourceSmallImageSize);
        dots.attr("height", resourceSmallImageSize);
        var lines = svg.selectAll(".lineTest");
        lines.attr("stroke-width", 2);
        // ths.attr("r", "7");
        ths.attr("width", resourceLargeImageSize);
        ths.attr("height", resourceLargeImageSize);
        // cloud_label.text(d.resourceId).style("fill", myColor(d.pathwayId));
        cloud_label.text(d.resourceName).style("fill", myColor(d.pathwayId));
        myWordCloud.update(tagMap[d.resourceId]);
        listResources([d]);
      }, 300);
    }
  })
  .on("dblclick", function (d) {
    t = d.pathwayLod - 0.5;
    if (
      !path_visible ||
      !all_visible ||
      zoomLevel > 6 ||
      (zoomLevel - 1 <= t && zoomLevel > t)
    ) {
      clearTimeout(timeout);
      toggleResource(d);
      if (d3.select(this).attr("marked") === "0") {
        d3.select(this).attr("stroke", "#404142");
        d3.select(this).attr("marked", "1");
      } else {
        d3.select(this).attr("stroke", "");
        d3.select(this).attr("marked", "0");
      }
    }
  })
  .style("fill", function (d) {
    return myColor(d.pathwayId);
  })
  .style("stroke-width", "5px")
  .attr("stroke", "")
  .attr("marked", "0");

var path_label = svg.selectAll(".path_label");

path_label
  .data(pathData)
  .enter()
  .append("text")
  .attr("class", "path_label")
  .attr("clip-path", "url(#clip)")
  .attr("x", 30) // shift the text a bit more right
  .attr("font-weight", 600)
  .text(function (d) {
    return d.values[0].pathwayId;
  })
  .style("fill", function (d) {
    return myColor(d.values[0].pathwayId);
  })
  .style("font-size", "15")
  .style("pointer-events", "none")
  .style("background-color", "#ffffff");

let newX, newY;

function updateChart() {
  // recover the new scale
  newX = d3.event.transform.rescaleX(x);
  newY = d3.event.transform.rescaleY(y);
  zoomLevel =
    (x.domain()[1] - x.domain()[0]) / (newX.domain()[1] - newX.domain()[0]);
  xAxis.call(d3.axisBottom(newX));
  yAxis.call(d3.axisLeft(newY));

  var lines = svg.selectAll(".lineTest");
  var sel = d3
    .line()
    .curve(d3.curveStepAfter) //https://github.com/d3/d3-shape/blob/master/README.md#curves //curveNatural
    .x((d) => newX(d.normX))
    .y((d) => newY(d.normY));

  lines.attr("d", (d) => sel(d.values));

  if (path_visible)
    lines.style("opacity", function (d) {
      t = d.values[0].pathwayLod - 0.5;
      if (
        !all_visible ||
        zoomLevel > 6 ||
        (zoomLevel - 1 <= t && zoomLevel > t)
      )
        return 1;
      else return 0.1;
    });

  var dots = svg.selectAll(".myDots");

  dots
    .attr("x", (d) => newX(d.normX) - resourceSmallImageSize / 2)
    .attr("y", (d) => newY(d.normY) - resourceSmallImageSize / 2);

  if (path_visible)
    dots.style("opacity", function (d) {
      t = d.pathwayLod - 0.5;
      if (
        !all_visible ||
        zoomLevel > 6 ||
        (zoomLevel - 1 <= t && zoomLevel > t)
      )
        return 0.8;
      else return 0.05;
    });

  /* CHANGING THE clickBubble on zoom */
  const clickBubble = d3.selectAll(".clickBubble");
  clickBubble.attr("cx", (d) => newX(d.normX)).attr("cy", (d) => newY(d.normY));

  /* CHANGING THE userPosition on zoom */
  const userPositionIcon = d3.selectAll(".userPositionIcon");
  userPositionIcon
    .attr("x", (d) => newX(d.position.x))
    .attr("y", (d) => newY(d.position.y));

  var detail;
  if (zoomLevel > 6) detail = 7;
  else detail = Math.floor(zoomLevel + 0.5);

  if (all_visible) LOD_label.text("Level of Detail:  " + detail.toString());

  if (!all_visible) {
    var path_label = svg.selectAll(".path_label");
    path_label
      .attr("dx", (d) => newX(d.values[d.values.length - 1].normX))
      .attr("dy", (d) => newY(d.values[d.values.length - 1].normY));
  }

  if (all_visible === true) {
    var path_label = svg.selectAll(".path_label");
    path_label
      .attr("dx", function (d) {
        t = d.values[0].pathwayLod - 0.5;
        if (zoomLevel - 1 <= t && zoomLevel > t)
          return newX(d.values[d.values.length - 1].normX);
        else return "-1000";
      })
      .attr("dy", function (d) {
        t = d.values[0].pathwayLod - 0.5;
        if (zoomLevel - 1 <= t && zoomLevel > t)
          return newY(d.values[d.values.length - 1].normY);
        else return "-1000";
      });
  }
}

function drawFilterPaths(duration) {
  var lines = svg.selectAll(".lineTest");

  lines
    .data(pathFilter)
    .enter()
    .append("path")
    .attr("class", "lineTest")
    .merge(lines)
    .transition()
    .duration(duration)
    .attr("d", (d) => sel(d.values))
    .attr("stroke-width", 2);
}

function drawFilterPathLabels(duration) {
  var path_label = svg.selectAll(".path_label");

  path_label
    .data(pathFilter)
    .enter()
    .append("text")
    .attr("class", "path_label")
    .merge(path_label)
    .transition()
    .duration(duration)
    .attr("dx", function (d) {
      if (all_visible && d.values[0].pathwayLod != 1) {
        return "-1000";
      }
      return x(d.values[d.values.length - 1].normX);
    })
    .attr("dy", function (d) {
      if (all_visible && d.values[0].pathwayLod != 1) {
        return "-1000";
      }
      return y(d.values[d.values.length - 1].normY);
    })
    .attr("selected", "0");
}

function drawFilterRess(duration) {
  var dots = svg.selectAll(".myDots");

  dots
    .data(resFilter)
    .enter()
    .append("circle")
    .attr("class", "myDots")
    .merge(dots)
    .transition()
    .duration(duration)
    .attr("x", (d) => x(d.normX) - resourceSmallImageSize / 2)
    .attr("y", (d) => y(d.normY) - resourceSmallImageSize / 2)
    .attr("width", resourceSmallImageSize)
    .attr("height", resourceSmallImageSize);
}

// A function that update the chart
function update(selectedGroup) {
  var w = 1;
  words = [];
  clearInterval(cycle);

  gElem.transition().duration(duration).call(zoom.transform, d3.zoomIdentity);

  pathFilter = [];
  resFilter = [];
  inView = selectedGroup.length;
  if (selectedGroup[0] === 0) {
    LOD_label.text("Level of Detail: 1");
    words = defaultWords[0];
    pathFilter = pathData;
    resFilter = resData;
    (rx = grx), (ry = gry), (sx = gsx), (sy = gsy);
    all_visible = true;
  } else {
    LOD_label.text("");
    pathFilter = pathTran.slice();
    sx = Number.MAX_VALUE;
    sy = Number.MAX_VALUE;
    rx = Number.MIN_VALUE;
    ry = Number.MIN_VALUE;
    for (var i = 0; i < selectedGroup.length; i++) {
      words.push(pathData[selectedGroup[i] - 1]);
      pathFilter[selectedGroup[i] - 1] = pathData[selectedGroup[i] - 1];
      sx = Math.min(
        sx,
        d3.min(pathFilter[selectedGroup[i] - 1].values, function (d) {
          return +d.normX;
        })
      );
      rx = Math.max(
        rx,
        d3.max(pathFilter[selectedGroup[i] - 1].values, function (d) {
          return +d.normX;
        })
      );
      sy = Math.min(
        sy,
        d3.min(pathFilter[selectedGroup[i] - 1].values, function (d) {
          return +d.normY;
        })
      );
      ry = Math.max(
        ry,
        d3.max(pathFilter[selectedGroup[i] - 1].values, function (d) {
          return +d.normY;
        })
      );
    }
    all_visible = false;
    for (var k = 0; k < pathFilter.length; k++) {
      resFilter = resFilter.concat(pathFilter[k].values);
    }
  }
  x.domain([sx - 1, rx + 1]);
  y.domain([sy - 1, ry + 1]);
  updateAxis(x, y, duration);

  drawFilterPaths(duration);
  drawFilterPathLabels(duration);
  drawFilterRess(duration);

  if (all_visible) {
    myWordCloud.update(mapCloud);
    cloud_label.text(mapName).style("fill", "");
    container = document.getElementById("listTitle");
    container.innerHTML = "Select a Resource or a Path";
    container = document.getElementById("resourceCards");
    container.innerHTML = "";
    for (var k = 0; k < 3; k++) {
      container.innerHTML += placeholder;
    }
  } else {
    myWordCloud.update(getTags(words[0]));
    cycle = setInterval(function () {
      // myWordCloud.update(getTags(words[w % words.length]));
      w++;
    }, 10000);
  }

  duration = 1000;
}

// setting up ui elements
var vals = [];
for (var k = 0; k < pathData.length; k++) {
  vals.push({
    name: pathData[k].values[0].pathwayId,
    value: k + 1
  });
}
sv = [0];
svt = [];
var clear;
$(".ui.multiple.search.selection.dropdown").dropdown({
  values: vals,
  placeholder: "All Paths",
  onChange: function (value, text, $selectedItem) {
    if (!clear) {
      sv = [];
      svt = value.split(",");
      for (var i = 0; i < svt.length; i++) {
        sv.push(+svt[i]);
      }
      update(sv);
    }
  },
});

const resetOnClick = () => {
  sv = [0];
  clear = true;
  $(".ui.multiple.search.selection.dropdown").dropdown("clear");
  clear = false;
  update(sv);
};

$(".button.reset").on("click", resetOnClick);

$(".button.clear").on("click", function () {
  container = document.getElementById("markedResources");
  container.innerHTML = "";
  markedSet.clear();
  var dots = svg.selectAll(".myDots");
  dots.attr("stroke", "").attr("marked", "0");
});

$(".ui.checkbox.path")
  .checkbox("set checked")
  .first()
  .checkbox({
    onChecked: function () {
      var lines = svg.selectAll(".lineTest");
      lines.style("opacity", function (d) {
        t = d.values[0].pathwayLod - 0.5;
        if (
          !all_visible ||
          zoomLevel > 6 ||
          (zoomLevel - 1 <= t && zoomLevel > t)
        )
          return 1;
        else return 0.1;
      });
      path_visible = true;
      var dots = svg.selectAll(".myDots");
      dots.style("opacity", function (d) {
        t = d.pathwayLod - 0.5;
        if (
          !all_visible ||
          zoomLevel > 6 ||
          (zoomLevel - 1 <= t && zoomLevel > t)
        )
          return 0.8;
        else return 0.05;
      });
    },
    onUnchecked: function () {
      var lines = svg.selectAll(".lineTest");
      lines.style("opacity", 0);
      var dots = svg.selectAll(".myDots");
      dots.style("opacity", 0.8);
      path_visible = false;
    },
  });

$(".ui.checkbox.plabels")
  .checkbox("set checked")
  .first()
  .checkbox({
    onChecked: function () {
      var path_label = svg.selectAll(".path_label");
      path_label.style("opacity", 0.9);
      plabel_visible = true;
    },
    onUnchecked: function () {
      var path_label = svg.selectAll(".path_label");
      path_label.style("opacity", 0);
      plabel_visible = false;
    },
  });

$(".menu .item").tab();

if (!path_avaliable) {
  $(".ui.checkbox.path").checkbox("uncheck");
  $(".ui.checkbox.path").checkbox("set disabled");
  $(".ui.checkbox.plabels").checkbox("uncheck");
  $(".ui.checkbox.plabels").checkbox("set disabled");
}