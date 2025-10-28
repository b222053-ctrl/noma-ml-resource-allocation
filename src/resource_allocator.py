# Resource Allocator Module

class ResourceAllocator:
    def __init__(self):
        # Initialize resources
        self.resources = {}

    def add_resource(self, resource_name, quantity):
        """Add a resource to the allocator."""
        if resource_name in self.resources:
            self.resources[resource_name] += quantity
        else:
            self.resources[resource_name] = quantity

    def allocate_resource(self, resource_name, quantity):
        """Allocate a specified quantity of a resource."""
        if resource_name in self.resources and self.resources[resource_name] >= quantity:
            self.resources[resource_name] -= quantity
            return True
        return False

    def release_resource(self, resource_name, quantity):
        """Release a specified quantity of a resource."""
        if resource_name in self.resources:
            self.resources[resource_name] += quantity

    def get_resources(self):
        """Get the current state of resources."""
        return self.resources
