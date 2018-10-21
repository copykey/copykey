keyway_name = "$keyway_name";
baseline = $baseline;
keypoints = $keypoints;

keyways = [
    ["master", 2, 7.25],
    ["schlage", 2.1, 9.1],
    ["kwikset", 1.8, 9.3]
];

function getWidth() = keyways[search([keyway_name], keyways)[0]][1];

echo(getWidth());

function getHeight() = keyways[search([keyway_name], keyways)[0]][2];


module key() {
    linear_extrude(height = getWidth())
    polygon(points = keypoints);
}

module keyway() {
    rotate([0, 90, 0]) translate([-1 * getWidth(), baseline, 0])
    linear_extrude(height = 1000000)
    import(str(keyway_name, ".dxf"));
}

module keyway_mask() {
    rotate([0, 90, 0])
    translate([-1 * getWidth(), baseline, 0])
    linear_extrude(height = 1000000)
    polygon(points = [[0, 0], [getWidth(), 0], [getWidth(), getHeight()], [0, getHeight()]]);
}

union() {
    intersection() {
        key();
        keyway();
    }
    difference() {
        key();
        keyway_mask();
    }
}